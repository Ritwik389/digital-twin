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

from chroma_utils import get_chroma_collection
from sentence_transformers import CrossEncoder
TOP_K = 15
TOP_K_FINAL = 5
INITIAL_FETCH = 15
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Short greetings / pleasantries — skip retrieval entirely
NO_RETRIEVAL_PATTERNS = [
    r"^(hi|hello|hey|thanks|thank you|okay|ok|sure|got it|interesting|cool|wow|nice|great|bye|goodbye|see you)\b",
    r"^(that'?s? (great|interesting|cool|helpful|fascinating|amazing))",
    r"^(i see|understood|makes sense|fair enough)",
]


def _is_no_retrieval(query: str) -> bool:
    q = query.lower().strip()
    return any(re.search(p, q) for p in NO_RETRIEVAL_PATTERNS)


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

    try:
        results = collection.query(**kwargs)
    except Exception:
        # If the metadata filter is too restrictive / errors, retry without it
        kwargs.pop("where", None)
        results = collection.query(**kwargs)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", ""),
            "year": meta.get("year", 0),
            "domain": meta.get("domain", ""),
            "filename": meta.get("filename", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": dist,
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
) -> dict:
    """
    Basic RAG entry point used by agent.py.

    Returns a dict with the retrieved context, the raw chunks, and a
    couple of flags the agent/UI use for display purposes.
    """
    if _is_no_retrieval(query):
        return {
            "context": "",
            "chunks": [],
            "query_type": "no_retrieval",
            "needs_retrieval": False,
        }

    chunks = retrieve_chunks(query, year_min=year_min, year_max=year_max, domain_filter=domain_filter)
    if chunks:
        # Create pairs of [query, document_text]
        pairs = [[query, c["text"]] for c in chunks]
        
        # Score the pairs using the local cross-encoder
        scores = reranker.predict(pairs)
        
        # Attach the scores to our chunk dictionaries
        for chunk, score in zip(chunks, scores):
            chunk["cross_score"] = float(score)
            
        # Sort chunks by the new cross-encoder score (highest to lowest)
        chunks.sort(key=lambda x: x["cross_score"], reverse=True)
        
        # Slice to keep only the absolute best matches
        chunks = chunks[:TOP_K_FINAL]
    context = build_context(chunks)

    return {
        "context": context,
        "chunks": chunks,
        "query_type": "direct" if chunks else "no_context",
        "needs_retrieval": bool(chunks),
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
