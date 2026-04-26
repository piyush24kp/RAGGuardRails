"""
Loads all department documents, chunks them, embeds with sentence-transformers,
and stores in ChromaDB with RBAC metadata (department, source filename).

Supports incremental updates: only files whose content has changed since the
last run are re-embedded. File hashes are stored in a sidecar JSON file.

Run once (or re-run anytime docs change):
    python -m ingestion.loader
"""
import os
import csv
import glob
import json
import hashlib
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, DATA_DIR

HASH_STORE = os.path.join(CHROMA_PERSIST_DIR, "file_hashes.json")


def _file_hash(path: str) -> str:
    """SHA-256 of the file contents — used to detect changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _load_hash_store() -> dict:
    if os.path.exists(HASH_STORE):
        with open(HASH_STORE) as f:
            return json.load(f)
    return {}


def _save_hash_store(store: dict) -> None:
    os.makedirs(os.path.dirname(HASH_STORE), exist_ok=True)
    with open(HASH_STORE, "w") as f:
        json.dump(store, f, indent=2)


def _load_md(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _load_csv(path: str) -> str:
    """Convert each CSV row to a readable sentence so the LLM can reason over it."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line = ", ".join(f"{k}: {v}" for k, v in row.items() if v)
            rows.append(line)
    return "\n".join(rows)


def load_documents(incremental: bool = True) -> tuple[list[dict], list[str]]:
    """
    Returns (changed_docs, deleted_sources).

    changed_docs: files that are new or have changed content since last run.
    deleted_sources: filenames that were in the hash store but no longer exist on disk
                     (their chunks should be removed from ChromaDB).

    Set incremental=False to force a full reload of every file.
    """
    hash_store = _load_hash_store() if incremental else {}
    new_hash_store = {}

    docs = []
    seen_sources = set()

    for dept_path in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
        if not os.path.isdir(dept_path):
            continue
        department = os.path.basename(dept_path)
        for filepath in sorted(glob.glob(os.path.join(dept_path, "*"))):
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in (".md", ".csv"):
                continue

            source = os.path.basename(filepath)
            seen_sources.add(source)
            current_hash = _file_hash(filepath)
            new_hash_store[source] = current_hash

            if incremental and hash_store.get(source) == current_hash:
                continue  # unchanged — skip

            text = _load_md(filepath) if ext == ".md" else _load_csv(filepath)
            docs.append({"text": text, "department": department, "source": source})

    # files that existed before but are now gone
    deleted = [src for src in hash_store if src not in seen_sources]

    _save_hash_store(new_hash_store)
    return docs, deleted


def chunk_documents(docs: list[dict], chunk_size: int = 800, chunk_overlap: int = 100) -> list[dict]:
    """Split each document into smaller chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        parts = splitter.split_text(doc["text"])
        for i, part in enumerate(parts):
            chunks.append({
                "text": part,
                "department": doc["department"],
                "source": doc["source"],
                "chunk_id": f"{doc['department']}__{doc['source']}__{i}",
            })
    return chunks


def _get_or_create_collection(client, ef):
    try:
        return client.get_collection(name=CHROMA_COLLECTION, embedding_function=ef)
    except Exception:
        return client.create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )


def update_vector_store(chunks: list[dict], deleted_sources: list[str]) -> None:
    """
    Incrementally update ChromaDB:
    - Delete all chunks belonging to removed/changed files (by source filename).
    - Upsert the new chunks for those files.
    This leaves unchanged files' chunks untouched.
    """
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    collection = _get_or_create_collection(client, ef)

    # sources that need their old chunks wiped (changed or deleted files)
    sources_to_clear = set(deleted_sources) | {c["source"] for c in chunks}
    for source in sources_to_clear:
        try:
            existing = collection.get(where={"source": source}, include=[])
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
                print(f"  Deleted {len(existing['ids'])} old chunks for '{source}'")
        except Exception:
            pass

    if not chunks:
        print("  No new/changed files to embed.")
    else:
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            collection.upsert(
                ids=[c["chunk_id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[{"department": c["department"], "source": c["source"]} for c in batch],
            )
            print(f"  Upserted chunks {i} – {i + len(batch) - 1}")

    print(f"\nVector store ready: {collection.count()} chunks in '{CHROMA_COLLECTION}'")


if __name__ == "__main__":
    import sys
    force_full = "--full" in sys.argv

    print("Scanning documents..." + (" (full rebuild)" if force_full else " (incremental)"))
    docs, deleted = load_documents(incremental=not force_full)

    if not docs and not deleted:
        print("  No changes detected. Vector store is up to date.")
    else:
        print(f"  Changed/new files: {[d['source'] for d in docs]}")
        print(f"  Deleted files: {deleted}")

        print("Chunking changed files...")
        chunks = chunk_documents(docs)
        print(f"  Produced {len(chunks)} chunks")

        print("Updating vector store...")
        update_vector_store(chunks, deleted)
