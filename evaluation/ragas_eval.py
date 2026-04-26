"""
Runs Ragas evaluation on a small hand-crafted test set.
Metrics: faithfulness, answer_relevancy, context_precision

Uses Groq (free) for the LLM and HuggingFace sentence-transformers for
embeddings — no OpenAI key required.

Run with:  python -m evaluation.ragas_eval
"""
import os
os.environ["RAGAS_DO_NOT_TRACK"] = "true"  # silence Ragas telemetry noise

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from retrieval.rag_chain import answer as rag_answer
from retrieval.vector_store import retrieve
from config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL

# ── Test set ──────────────────────────────────────────────────────────────────
TEST_SET = [
    {
        "role": "finance",
        "question": "What was the revenue growth percentage in 2024?",
        "ground_truth": "Revenue grew by 25% in 2024.",
    },
    {
        "role": "hr",
        "question": "What is the company's leave policy?",
        "ground_truth": "Employees are entitled to annual, sick, and casual leave as described in the employee handbook.",
    },
    {
        "role": "ceo",
        "question": "What are the key financial ratios for 2024?",
        "ground_truth": "Gross margin was 60%, up from 55% in 2023.",
    },
]


def run_eval():
    # Use Groq LLM — same model as the app, no extra API key needed
    llm = LangchainLLMWrapper(
        ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    )

    # Use local HuggingFace embeddings — no OpenAI key needed
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    print("Collecting answers from RAG pipeline...")
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_SET:
        print(f"  Q ({item['role']}): {item['question']}")
        result = rag_answer(item["question"], item["role"])
        chunks = retrieve(item["question"], item["role"])

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([c["text"] for c in chunks])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nRunning Ragas evaluation...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n── Ragas Evaluation Results ──")
    print(results)
    return results


if __name__ == "__main__":
    run_eval()
