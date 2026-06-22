from pathlib import Path

from chroma_utils import get_chroma_collection


CLEAN_DIR = Path("data/cleaned")


def parse_metadata_header(content: str) -> tuple[dict, str]:
    parts = content.split("---\n", 1)
    if len(parts) != 2:
        return {}, content
    header, body = parts
    meta = {}
    for line in header.strip().split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            meta[key.strip().lower()] = val.strip()
    if "year" in meta:
        try:
            meta["year"] = int(meta["year"])
        except ValueError:
            meta["year"] = 0
    return meta, body.strip()


def already_ingested(collection, filename: str) -> bool:
    results = collection.get(where={"filename": filename}, limit=1)
    return len(results["ids"]) > 0





def ingest_file(filepath: Path, collection):
    if already_ingested(collection, filepath.name):
        print(f"[skip] {filepath.name} already in ChromaDB")
        return

    content = filepath.read_text(encoding="utf-8")
    meta, body = parse_metadata_header(content)
    from semantic_chunker import SemanticChunker




    if not body:
        print(f"[skip] {filepath.name} — empty body after header parse")
        return
    chunker = SemanticChunker(
        similarity_threshold=0.2,  
        min_chunk_size=2000,
        max_chunk_size=6000
    )

    chunks = chunker.chunk(body)

    print(f"Document split into {len(chunks)} semantic chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print()

    print(f"[ingest] {filepath.name} -> {len(chunks)} semantic chunks")

    ids = [f"{filepath.stem}-{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": meta.get("source", filepath.stem),
            "year": meta.get("year", 0),
            "domain": meta.get("domain", "general"),
            "chunk_index": i,
            "filename": filepath.name,
        }
        for i in range(len(chunks))
    ]

    # documents=chunks -> ChromaDB embeds them locally with the
    # collection's embedding function (offline ONNX model, no API calls).
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )

    print(f"[done] {filepath.name} - {len(chunks)} chunks stored")


def ingest_all():
    collection = get_chroma_collection()
    print(f"[chroma] existing docs: {collection.count()}")

    txt_files = sorted(CLEAN_DIR.glob("*.txt"))
    if not txt_files:
        print("[warn] No cleaned .txt files found in data/cleaned/")
        print("       Run cleaner.py first, or manually add .txt files with metadata headers.")
        return

    for f in txt_files:
        ingest_file(f, collection)

    print(f"\n[done] Total chunks in ChromaDB: {collection.count()}")


if __name__ == "__main__":
    ingest_all()
