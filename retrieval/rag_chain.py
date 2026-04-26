"""
Builds the answer given a user query and their role.
Flow: query → RBAC-filtered retrieval → prompt assembly → Groq LLM → answer
"""
from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL
from retrieval.vector_store import retrieve

SYSTEM_PROMPT = """You are an internal company assistant for FinSolve Technologies.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say "I don't have enough information to answer that."
Do not make up facts. Be concise and professional."""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source']} | Dept: {c['department']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer(query: str, role: str) -> dict:
    """
    Returns {answer, sources, chunks_used}.
    sources is a list of unique filenames used to build the answer.
    """
    chunks = retrieve(query, role)

    if not chunks:
        return {
            "answer": "I couldn't find any relevant information for your query in documents you have access to.",
            "sources": [],
            "chunks_used": 0,
        }

    context = build_context(chunks)
    sources = list({c["source"] for c in chunks})

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": sources,
        "chunks_used": len(chunks),
    }
