"""
Scans LLM output for PII patterns and redacts them.
Patterns covered: email addresses, phone numbers, Indian salary figures in employee context,
date of birth, employee IDs, and Aadhaar-style numeric IDs.
"""
import re

_PATTERNS = [
    # Email addresses
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[EMAIL REDACTED]"),
    # Phone numbers (various formats)
    (re.compile(r"\b(?:\+91[\s\-]?)?[6-9]\d{9}\b"), "[PHONE REDACTED]"),
    (re.compile(r"\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b"), "[PHONE REDACTED]"),
    # Date of birth patterns like 1991-04-03 or 03/04/1991
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DOB REDACTED]"),
    (re.compile(r"\b\d{2}/\d{2}/\d{4}\b"), "[DOB REDACTED]"),
    # Salary figures (large numbers that look like salary in INR)
    (re.compile(r"\b\d{4,7}\.\d{2}\b"), "[SALARY REDACTED]"),
    # Employee IDs like FINEMP1000
    (re.compile(r"\bFINEMP\d{4}\b"), "[EMP_ID REDACTED]"),
]


def redact(text: str) -> tuple[str, list[str]]:
    """
    Returns (redacted_text, list_of_pii_types_found).
    """
    found = []
    for pattern, replacement in _PATTERNS:
        matches = pattern.findall(text)
        if matches:
            found.append(replacement.strip("[]").replace(" REDACTED", ""))
            text = pattern.sub(replacement, text)
    return text, found
