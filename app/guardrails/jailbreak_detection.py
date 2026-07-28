"""
app/guardrails/jailbreak_detection.py

Heuristic jailbreak-attempt detection on the user's query.

Distinct from prompt_injection.py:
  - prompt_injection.py detects attempts to override the system's
    instructions, including via retrieved document content.
  - This module detects attempts, from the user directly, to get the
    system to roleplay, hypothesize, or "act as" something that bypasses
    its intended purpose (document Q&A) and normal behavior -- e.g. classic
    "DAN"/"do anything now" style prompts, fictional-framing bypasses, or
    persona-swap requests.

Same heuristic/pattern-based approach and same conservative bias as
prompt_injection.py -- flags rather than silently blocks; the caller
(chat_stream_routes.py) decides what to do with a flagged result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_JAILBREAK_PATTERNS = [
    r"\bDAN\b",
    r"do anything now",
    r"developer mode",
    r"jailbreak(ed)?",
    r"no (restrictions|filters|limits|rules) (apply|mode)",
    r"without (any )?(restrictions|filters|limitations|guidelines)",
    r"unrestricted (mode|version|ai)",
    r"hypothetically,? (if )?you (had no|could ignore)",
    r"for (a )?(fictional|hypothetical) (story|scenario|purpose)s?,? (ignore|bypass|disregard)",
    r"pretend (there are|there is) no (rules|restrictions|guidelines)",
    r"respond (only )?as .*(character|persona|entity) (with no|without)",
    r"two (ais?|models?|personas?),? one (restricted|normal).*(one|another) (unrestricted|free)",
    r"opposite day",
    r"evil (twin|version|mode)",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _JAILBREAK_PATTERNS]


@dataclass
class JailbreakCheckResult:
    is_suspicious: bool
    matched_patterns: list[str] = field(default_factory=list)


def detect(query: str) -> JailbreakCheckResult:
    if not query:
        return JailbreakCheckResult(is_suspicious=False)

    matched = [p.pattern for p in _COMPILED_PATTERNS if p.search(query)]
    return JailbreakCheckResult(is_suspicious=bool(matched), matched_patterns=matched)