"""
Runs Ragas evaluation on a small hand-crafted test set.
Metrics: faithfulness, answer_relevancy, context_precision

Run with:  python -m evaluation.ragas_eval
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

from retrieval.rag_chain import answer as rag_answer
from retrieval.vector_store import retrieve
from config import GROQ_API_KEY, GROQ_MODEL

# ── Test set ──────────────────────────────────────────────────────────────────
# Each entry: question asked as a specific role, plus the ground-truth answer
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
    llm = LangchainLLMWrapper(
        ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    )

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_SET:
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

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
    )

    print("\n── Ragas Evaluation Results ──")
    print(results)
    return results


if __name__ == "__main__":
    run_eval()
