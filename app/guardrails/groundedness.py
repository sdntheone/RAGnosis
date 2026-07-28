"""
app/guardrails/groundedness.py

Fast, LLM-free groundedness scoring: how much of the generated answer is
lexically supported by the retrieved chunks it was supposed to be based on.

This runs on every response (cheap n-gram overlap, no extra API call) and
feeds the "confidence indicator" shown on the observability dashboard
(observability/*). It is intentionally a heuristic, not a semantic judge --
hallucination_check.py (next file) is the semantic, LLM-as-judge version,
meant to run selectively (e.g. only when this score is low) since it costs
an extra generation call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class GroundednessResult:
    score: float  # 0.0 (no overlap) to 1.0 (fully covered by context)
    covered_ngram_count: int
    total_ngram_count: int


def score_groundedness(answer: str, context_chunks: list[str], ngram_size: int = 3) -> GroundednessResult:
    """Fraction of the answer's n-grams that also appear somewhere in the
    retrieved context. A low score means the answer is introducing content
    the retrieved chunks don't actually contain -- a strong hallucination
    signal, computed without an extra LLM call.
    """
    answer_ngrams = _ngrams(answer, ngram_size)
    if not answer_ngrams:
        return GroundednessResult(score=1.0, covered_ngram_count=0, total_ngram_count=0)

    context_ngrams: set[tuple] = set()
    for chunk in context_chunks:
        context_ngrams |= _ngrams(chunk, ngram_size)

    covered = answer_ngrams & context_ngrams
    score = len(covered) / len(answer_ngrams)

    return GroundednessResult(
        score=round(score, 4),
        covered_ngram_count=len(covered),
        total_ngram_count=len(answer_ngrams),
    )


def _ngrams(text: str, n: int) -> set[tuple]:
    words = _WORD_PATTERN.findall(text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}