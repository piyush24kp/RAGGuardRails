"""
Detects whether a question is relevant to company business.
Uses a lightweight LLM call to classify intent.
Returns (is_in_scope: bool, reason: str).
"""
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

_CLASSIFIER_PROMPT = """You are a strict input classifier for a company internal chatbot.
Your job is to decide if the user's question is related to company business topics such as:
- HR policies, leave, attendance, payroll, employee data
- Financial reports, budgets, expenses, revenue
- Marketing reports and campaigns
- Engineering or product documentation
- General company policies and procedures

Reply with exactly one word: IN_SCOPE or OUT_OF_SCOPE.
Do not explain. Do not add punctuation."""

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def check_scope(query: str) -> tuple[bool, str]:
    """Returns (True, '') if in scope, (False, reason) if out of scope."""
    query = query.strip()
    if not query:
        return False, "Empty question."

    response = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _CLASSIFIER_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    verdict = response.choices[0].message.content.strip().upper()
    if verdict == "IN_SCOPE":
        return True, ""
    return False, "Your question doesn't appear to be related to company business. Please ask about HR, finance, engineering, or company policies."
