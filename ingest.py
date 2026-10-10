from pathlib import Path

from chroma_utils import get_chroma_collection
import hashlib
from source_utils import provenance,effective_year

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





def ingest_file(filepath: Path, collection, chunker):
    content=filepath.read_text(encoding='utf-8')
    meta,body=parse_metadata_header(content)
    if not body:
        print(f'[skip] {filepath.name}: empty source body; previous chunks retained')
        return
    version=hashlib.sha256(('semantic-v2\0'+content).encode()).hexdigest()
    previous=collection.get(where={'filename':filepath.name},include=['metadatas'])
    if previous['ids'] and all(item.get('corpus_version')==version for item in previous['metadatas']):
        print(f'[skip] {filepath.name}: content version unchanged')
        return
    chunks=chunker.chunk(body)
    if not chunks:
        print(f'[skip] {filepath.name}: no chunks produced; previous chunks retained')
        return
    ids=[f'{filepath.stem}-{version[:16]}-{i}' for i in range(len(chunks))]
    source=provenance(meta | {'filename':filepath.name})
    metadata={'source':meta.get('source',filepath.stem),'year':effective_year(meta | {'filename':filepath.name}),
              'domain':meta.get('domain','general'),'filename':filepath.name,'corpus_version':version}
    if source['url']:metadata['url']=source['url']
    # File-level headers do not locate individual transcript chunks in a video.
    metadatas=[metadata | {'chunk_index':i} for i in range(len(chunks))]
    collection.upsert(ids=ids,documents=chunks,metadatas=metadatas)
    obsolete=[key for key in previous['ids'] if key not in ids]
    if obsolete:collection.delete(ids=obsolete)
    print(f'[done] {filepath.name}: {len(chunks)} current chunks; {len(obsolete)} outdated chunks removed')


def ingest_all():
    collection = get_chroma_collection()
    print(f"[chroma] existing docs: {collection.count()}")

    txt_files = sorted(CLEAN_DIR.glob("*.txt"))
    if not txt_files:
        print("[warn] No cleaned .txt files found in data/cleaned/")
        print("       Run cleaner.py first, or manually add .txt files with metadata headers.")
        return
    print("[ingest] Loading SemanticChunker model...")
    from semantic_chunker import SemanticChunker
    chunker = SemanticChunker(
        similarity_threshold=0.2,  
        min_chunk_size=2000,
        max_chunk_size=6000
    )

    for f in txt_files:
        ingest_file(f, collection, chunker)

    print(f"\n[done] Total chunks in ChromaDB: {collection.count()}")


if __name__ == "__main__":
    ingest_all()
