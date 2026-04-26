"""
Loads all department documents, chunks them, embeds with sentence-transformers,
and stores in ChromaDB with RBAC metadata (department, source filename).

Run once (or re-run to refresh the vector store):
    python -m ingestion.loader
"""
import os
import csv
import glob
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, DATA_DIR


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


def load_documents() -> list[dict]:
    """
    Returns a list of dicts: {text, department, source}
    One dict per file found under DATA_DIR/{department}/.
    """
    docs = []
    for dept_path in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
        if not os.path.isdir(dept_path):
            continue
        department = os.path.basename(dept_path)
        for filepath in sorted(glob.glob(os.path.join(dept_path, "*"))):
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".md":
                text = _load_md(filepath)
            elif ext == ".csv":
                text = _load_csv(filepath)
            else:
                continue  # skip unknown file types
            docs.append({
                "text": text,
                "department": department,
                "source": os.path.basename(filepath),
            })
    return docs


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


def build_vector_store(chunks: list[dict]) -> None:
    """Embed chunks and upsert into ChromaDB."""
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # delete existing collection so re-runs start fresh
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

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
    print("Loading documents...")
    docs = load_documents()
    print(f"  Found {len(docs)} files across departments: {sorted({d['department'] for d in docs})}")

    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"  Produced {len(chunks)} chunks")

    print("Building vector store...")
    build_vector_store(chunks)
