"""
chroma_utils.py
Shared ChromaDB collection helper.

Uses ChromaDB's built-in offline embedding function (a small ONNX-based
all-MiniLM-L6-v2 model, runs locally via onnxruntime). The model is
downloaded once on first use and cached locally — no Gemini / API calls
are made for embeddings.

Both ingest.py (writing chunks) and rag.py (querying chunks) import
get_chroma_collection() from here so they always use the SAME embedding
function, which is required for similarity search to work correctly.
"""

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "./storage/chroma_db"
COLLECTION_NAME = "jensen_rag_corpus"

_embedding_function = None
_collection = None


def get_embedding_function():
    """Return the shared offline embedding function (ONNX MiniLM, local)."""
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return _embedding_function


def get_chroma_collection():
    """Return the persistent ChromaDB collection, creating it if needed."""
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection
