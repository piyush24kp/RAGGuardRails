"""
Wraps ChromaDB and returns a role-filtered retriever.
The filter restricts results to departments the user's role can access.
"""
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL
from retrieval.rbac import get_allowed_departments


def _get_collection():
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=CHROMA_COLLECTION, embedding_function=ef)


def retrieve(query: str, role: str, n_results: int = 5) -> list[dict]:
    """
    Query ChromaDB with a department filter derived from the user's role.
    Returns a list of {text, department, source, distance}.
    """
    allowed = get_allowed_departments(role)
    if not allowed:
        return []

    collection = _get_collection()

    # ChromaDB $in filter
    where = {"department": {"$in": allowed}} if len(allowed) > 1 else {"department": allowed[0]}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "department": meta["department"],
            "source": meta["source"],
            "distance": round(dist, 4),
        })
    return chunks
