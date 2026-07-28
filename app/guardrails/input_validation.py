"""
app/guardrails/input_validation.py

Basic input validation -- the cheapest guardrail, run first, before any
retrieval or LLM call happens. Catches malformed, empty, oversized, or
control-character-laden input.

This is deliberately NOT where prompt-injection/jailbreak detection lives
(see prompt_injection.py, jailbreak_detection.py) -- this module only
checks structural validity, not intent. Keeping them separate means each
guardrail can be toggled/tuned independently and the overall guardrail
pipeline (wired in chat_stream_routes.py) can short-circuit cheaply on
basic invalid input before paying for a more expensive intent check.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MIN_QUERY_LENGTH = 1
MAX_QUERY_LENGTH = 4000  # generous for interview-prep style multi-part questions

# Control characters other than newline/tab, which can be used to smuggle
# hidden instructions or break downstream formatting.
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Repeated-character spam, e.g. "aaaaaaaaaaaaaaaaaaaa...", often used to
# probe context-window handling or pad past a filter.
_EXCESSIVE_REPEAT_PATTERN = re.compile(r"(.)\1{50,}")


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str | None = None
    cleaned_query: str | None = None


def validate_query(raw_query: str) -> ValidationResult:
    if raw_query is None:
        return ValidationResult(is_valid=False, reason="Query is missing.")

    query = unicodedata.normalize("NFKC", raw_query).strip()

    if len(query) < MIN_QUERY_LENGTH:
        return ValidationResult(is_valid=False, reason="Query is empty.")

    if len(query) > MAX_QUERY_LENGTH:
        return ValidationResult(
            is_valid=False,
            reason=f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters.",
        )

    if _CONTROL_CHAR_PATTERN.search(query):
        query = _CONTROL_CHAR_PATTERN.sub("", query)

    if _EXCESSIVE_REPEAT_PATTERN.search(query):
        return ValidationResult(is_valid=False, reason="Query contains excessive repeated characters.")

    if not query:
        return ValidationResult(is_valid=False, reason="Query is empty after cleaning.")

    return ValidationResult(is_valid=True, cleaned_query=query)


def validate_filename(filename: str, max_length: int = 255) -> ValidationResult:
    if not filename or not filename.strip():
        return ValidationResult(is_valid=False, reason="Filename is empty.")

    if len(filename) > max_length:
        return ValidationResult(is_valid=False, reason="Filename is too long.")

    if "/" in filename or "\\" in filename or ".." in filename:
        return ValidationResult(is_valid=False, reason="Filename contains invalid path characters.")

    return ValidationResult(is_valid=True, cleaned_query=filename.strip())