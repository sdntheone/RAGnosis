"""
app/guardrails/retrieval_validation.py

Validates retrieved chunks BEFORE generation -- catches the case where
retrieval returns technically-ranked-highest results that are still too
weak/irrelevant to answer the query from, so the system can respond with
"I don't have enough information in the uploaded documents" instead of
generating an answer from thin context (which groundedness.py and
hallucination_check.py would only catch AFTER the fact, at higher cost).

Two checks:
  1. similarity floor -- top result's similarity score must clear a
     minimum threshold
  2. minimum chunk count -- at least N chunks must be returned at all
     (an empty or near-empty retrieval is itself a signal to short-circuit)

Similarity scores are backend-specific (FAISS returns L2 distance by
default via similarity_search_with_score, not the plain
similarity_search used elsewhere in this codebase) -- this module accepts
scores as plain floats so it works with whatever the caller's backend
returns, as long as the caller passes similarity (higher = better), not
distance. See _normalize_note below for the FAISS distance->similarity
conversion callers should apply first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_SIMILARITY_SCORE = 0.38  # on a 0-1 cosine-similarity scale
MIN_CHUNK_COUNT = 1
LOW_CONFIDENCE_SIMILARITY_SCORE = 0.45  # below this, flag as "low confidence" but don't block


@dataclass
class RetrievalValidationResult:
    is_valid: bool
    confidence: str  # "high" | "medium" | "low" | "none"
    reason: str | None = None
    top_score: float | None = None
    chunk_count: int = 0
    scores: list[float] = field(default_factory=list)


def validate_retrieval(
    scores: list[float],
    min_similarity: float = MIN_SIMILARITY_SCORE,
    min_chunks: int = MIN_CHUNK_COUNT,
) -> RetrievalValidationResult:
    """`scores` must be similarity scores (higher = better match), already
    normalized to a comparable scale by the caller -- e.g. for FAISS's
    similarity_search_with_score (which returns L2 distance, lower = better),
    convert first with something like `similarity = 1 / (1 + distance)`.
    """
    if not scores:
        return RetrievalValidationResult(
            is_valid=False, confidence="none", reason="No chunks retrieved.", chunk_count=0
        )

    if len(scores) < min_chunks:
        return RetrievalValidationResult(
            is_valid=False,
            confidence="none",
            reason=f"Only {len(scores)} chunk(s) retrieved, below minimum of {min_chunks}.",
            top_score=max(scores),
            chunk_count=len(scores),
            scores=scores,
        )

    top_score = max(scores)

    if top_score < min_similarity:
        return RetrievalValidationResult(
            is_valid=False,
            confidence="none",
            reason=(
                f"Top similarity score {top_score:.3f} is below the minimum "
                f"threshold {min_similarity:.3f} -- retrieved content is likely "
                f"unrelated to the query."
            ),
            top_score=top_score,
            chunk_count=len(scores),
            scores=scores,
        )

    confidence = (
        "high" if top_score >= LOW_CONFIDENCE_SIMILARITY_SCORE + 0.2
        else "medium" if top_score >= LOW_CONFIDENCE_SIMILARITY_SCORE
        else "low"
    )

    return RetrievalValidationResult(
        is_valid=True,
        confidence=confidence,
        top_score=top_score,
        chunk_count=len(scores),
        scores=scores,
    )


def faiss_distance_to_similarity(distance: float) -> float:
    """Convert FAISS L2 distance (lower = better, unbounded) to a
    0-1 similarity score (higher = better) for use with validate_retrieval.
    """
    return 1.0 / (1.0 + max(distance, 0.0))