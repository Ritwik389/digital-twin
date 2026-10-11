"""
rag.py
Basic RAG retrieval.

- Embeds the query with the same offline ONNX model used at ingestion
  time (via chroma_utils) and pulls the top-k most similar chunks from
  ChromaDB.
- Simple greeting/social messages are detected with a cheap regex check
  so we don't waste a retrieval pass (and so the prompt doesn't get
  cluttered with irrelevant context) — this costs zero API calls.
"""

import re
import math
import time

from chroma_utils import get_chroma_collection
from source_utils import effective_year
import logging
import threading
TOP_K = 15
TOP_K_FINAL = 5
INITIAL_FETCH = 15
logger = logging.getLogger(__name__)
_reranker = None
_reranker_failed = False
_reranker_lock = threading.Lock()


class RerankerUnavailable(RuntimeError):
    """Evaluation must not misreport vector fallback as cross-encoder results."""


def _get_reranker():
    global _reranker, _reranker_failed
    with _reranker_lock:
        if _reranker is None and not _reranker_failed:
            try:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
            except Exception:
                logger.exception('Reranker unavailable; using embedding ranking')
                _reranker_failed = True
        return _reranker

# Short greetings / pleasantries — skip retrieval entirely
NO_RETRIEVAL_PATTERNS = [
    r"(?:hi|hello|hey)(?:[ ,]+jensen)?[!. ]*",
    r"(?:thanks|thank you|okay|ok|sure|got it|interesting|cool|wow|nice|great|bye|goodbye|see you)[!. ]*",
    r"(?:that'?s? (?:great|interesting|cool|helpful|fascinating|amazing)|i see|understood|makes sense|fair enough)[!. ]*",
]


def _is_no_retrieval(query: str) -> bool:
    return any(re.fullmatch(pattern, query.lower().strip()) for pattern in NO_RETRIEVAL_PATTERNS)


def retrieve_chunks(
    query: str,
    year_min: int | None = None,
    year_max: int | None = None,
    domain_filter: str | None = None,
    top_k: int = INITIAL_FETCH,
) -> list[dict]:
    """Return the top-k most relevant chunks for `query` from ChromaDB."""
    collection = get_chroma_collection()
    if collection.count() == 0:
        return []
    
    where_conditions = []
    if year_min is None and year_max is not None: year_min = 1
    if year_min is not None:
        where_conditions.append({"year": {"$gte": year_min}})
    if year_max is not None:
        where_conditions.append({"year": {"$lte": year_max}})
    if domain_filter is not None:
        where_conditions.append({"domain": {"$eq": domain_filter}})
    if len(where_conditions) == 1:
        where_clause = where_conditions[0]
    elif len(where_conditions) > 1:
        where_clause = {"$and": where_conditions}
    else:
        where_clause = None


    kwargs = {
        "query_texts": [query],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        kwargs["where"] = where_clause

    # A failed filtered query must not silently leak sources from another era.
    results = collection.query(**kwargs)

    chunks = []
    for position, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        meta = meta or {}
        year=effective_year(meta)
        if year_min is not None and year < year_min:continue
        if year_max is not None and (year <= 0 or year > year_max):continue
        if domain_filter is not None and meta.get('domain') != domain_filter:continue
        ids = results.get('ids', [[]])[0]
        chunks.append({
            "chunk_id": ids[position] if position < len(ids) else str(meta.get('chunk_index', '')),
            "text": doc,
            "source": meta.get("source", ""),
            "year": year,
            "domain": meta.get("domain", ""),
            "filename": meta.get("filename", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": dist,
            "url": meta.get("url"), "start_seconds": meta.get("start_seconds"),
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    parts = []
    for c in chunks:
        label = f"[{c['source']}, {c['year']}]"
        parts.append(f"{label}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def retrieve(
    query: str,
    year_min: int | None = None,
    year_max: int | None = None,
    domain_filter: str | None = None,
    conversation_history: str = "",
    strategy: str = "reranked",
    strict_reranker: bool = False,
    top_k: int = TOP_K_FINAL,
) -> dict:
    """
    Basic RAG entry point used by agent.py.

    Returns a dict with the retrieved context, the raw chunks, and a
    couple of flags the agent/UI use for display purposes.
    """
    if strategy not in {'vector', 'reranked'}:
        raise ValueError('Unknown retrieval strategy')
    if not 1 <= top_k <= INITIAL_FETCH:
        raise ValueError('top_k must be between 1 and the candidate-pool size')
    if _is_no_retrieval(query):
        return {
            "context": "",
            "chunks": [],
            "query_type": "no_retrieval",
            "needs_retrieval": False,
            "ranking_backend": "skipped", "candidate_count": 0, "candidate_ids": [],
            "candidate_ms": 0, "ranking_ms": 0,
        }

    # Resolve short follow-ups using the last user topic, without an extra model call.
    retrieval_query = query
    if conversation_history and re.search(r"\b(that|it|this|they|those|more|why)\b", query.lower()) and len(query.split()) < 14:
        previous = re.findall(r"^User: (.+)$", conversation_history, re.M)
        if previous: retrieval_query = previous[-1][-500:] + "\nFollow-up: " + query
    start = time.perf_counter()
    chunks = retrieve_chunks(retrieval_query, year_min=year_min, year_max=year_max, domain_filter=domain_filter)
    candidate_ms = (time.perf_counter() - start) * 1000
    candidate_ids = [c['chunk_id'] for c in chunks]
    ranking_backend = 'vector' if chunks else 'empty'
    start = time.perf_counter()
    reranker = _get_reranker() if chunks and strategy == 'reranked' else None
    if chunks and strategy == 'reranked' and reranker is None:
        ranking_backend = 'vector_fallback'
        if strict_reranker:
            raise RerankerUnavailable('The cross-encoder is unavailable; this is not a reranked evaluation.')
    if chunks and reranker:
        # Create pairs of [query, document_text]
        pairs = [[retrieval_query, c["text"]] for c in chunks]
        
        # Score the pairs using the local cross-encoder
        try:
            with _reranker_lock:
                scores = reranker.predict(pairs)
            if len(scores) != len(chunks) or not all(math.isfinite(float(s)) for s in scores):
                raise ValueError('Invalid reranker scores')
            ranking_backend = 'cross_encoder'
        except Exception:
            logger.exception('Reranking failed; retaining embedding ranking')
            if strict_reranker:
                raise RerankerUnavailable('The cross-encoder failed; reranked evaluation was not completed.') from None
            ranking_backend = 'vector_fallback'
            scores = [-c['distance'] for c in chunks]
        
        # Attach the scores to our chunk dictionaries
        for chunk, score in zip(chunks, scores):
            chunk["cross_score"] = float(score)
            
        # Sort chunks by the new cross-encoder score (highest to lowest)
        chunks.sort(key=lambda x: x["cross_score"], reverse=True)
        
        # Slice to keep only the absolute best matches
    ranking_ms = (time.perf_counter() - start) * 1000
    chunks = chunks[:top_k]
    context = build_context(chunks)

    return {
        "context": context,
        "chunks": chunks,
        "query_type": "direct" if chunks else "no_context",
        "retrieval_query": retrieval_query,
        "needs_retrieval": bool(chunks),
        "ranking_backend": ranking_backend, "candidate_ids": candidate_ids,
        "candidate_count": len(candidate_ids),
        "candidate_ms": round(candidate_ms, 2), "ranking_ms": round(ranking_ms, 2),
    }


if __name__ == "__main__":
    tests = [
        "Hey thanks, that was really helpful!",
        "What is CUDA?",
        "How did the AlexNet moment change NVIDIA's roadmap?",
    ]
    for q in tests:
        print(f"\n{'=' * 60}")
        print(f"Query: {q}")
        result = retrieve(q)
        print(f"query_type: {result['query_type']}")
        print(f"needs_retrieval: {result['needs_retrieval']}")
        print(f"chunks: {len(result['chunks'])}")
        print(f"context length: {len(result['context'])}")
