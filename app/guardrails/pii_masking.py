"""
app/guardrails/pii_masking.py

Regex-based PII detection and masking.

Used in two places (wired in chat_stream_routes.py):
  1. Before logging a query/response/retrieved-chunk to the observability
     layer (observability/*) -- so raw PII never lands in logs/traces.
  2. Optionally on retrieved chunk content before it's sent to the LLM,
     if a session is configured to redact PII from source documents
     (e.g. uploaded resumes, ID scans) -- off by default, since for most
     interview-prep use cases the person WANTS their own PII (e.g. their
     own resume's contact info) preserved in the answer.

This is regex-based, not an NER model -- fast, dependency-free, catches the
common structured PII types (email, phone, SSN-like, credit-card-like,
IP address). It will miss unstructured PII (a name in running text) --
documented as a known limitation rather than silently overclaiming
coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

_MASK_TOKEN = {
    "email": "[EMAIL_REDACTED]",
    "phone": "[PHONE_REDACTED]",
    "ssn": "[SSN_REDACTED]",
    "credit_card": "[CARD_REDACTED]",
    "ip_address": "[IP_REDACTED]",
}


@dataclass
class PiiMaskResult:
    masked_text: str
    found_types: list[str] = field(default_factory=list)
    match_count: int = 0


def mask(text: str, types: list[str] | None = None) -> PiiMaskResult:
    """Mask PII in `text`. If `types` is given, only those PII types are
    checked/masked (e.g. types=["email", "phone"]); otherwise all known
    types are checked.
    """
    if not text:
        return PiiMaskResult(masked_text=text or "", found_types=[], match_count=0)

    active_patterns = (
        {k: v for k, v in _PATTERNS.items() if k in types} if types else _PATTERNS
    )

    masked_text = text
    found_types = []
    match_count = 0

    for pii_type, pattern in active_patterns.items():
        matches = pattern.findall(masked_text)
        if matches:
            found_types.append(pii_type)
            match_count += len(matches)
            masked_text = pattern.sub(_MASK_TOKEN[pii_type], masked_text)

    return PiiMaskResult(masked_text=masked_text, found_types=found_types, match_count=match_count)


def contains_pii(text: str, types: list[str] | None = None) -> bool:
    if not text:
        return False
    active_patterns = (
        {k: v for k, v in _PATTERNS.items() if k in types} if types else _PATTERNS
    )
    return any(pattern.search(text) for pattern in active_patterns.values())