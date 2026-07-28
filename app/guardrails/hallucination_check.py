"""
app/guardrails/hallucination_check.py

LLM-as-judge hallucination detection: identifies specific claims in the
generated answer that are NOT supported by the retrieved context.

More expensive and more accurate than groundedness.py's n-gram overlap --
intended to run selectively (e.g. chat_stream_routes.py only calls this
when groundedness.score_groundedness() comes back low, or when a session
is running in a "strict" verification mode), not on every single response.

Reuses the same ChatOpenAI setup pattern as app/llm/rag_chain.py and
app/extraction/caption_generator.py -- same provider/env var, just a
separate low-temperature judge call with a structured-output prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

JUDGE_MODEL = "gpt-4o-mini"

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict fact-checking judge. You will be given a CONTEXT "
    "(retrieved document excerpts) and an ANSWER generated from that "
    "context. Identify any claims in the ANSWER that are not supported by "
    "the CONTEXT -- these are hallucinations. "
    "Respond ONLY with JSON, no other text, in this exact shape: "
    '{"unsupported_claims": ["claim 1", "claim 2"], '
    '"is_grounded": true or false, '
    '"confidence": "high" or "medium" or "low"}. '
    'If every claim in the ANSWER is supported by the CONTEXT, return '
    '"unsupported_claims": [] and "is_grounded": true.'
)

_judge_llm = None


def _get_judge_llm() -> ChatOpenAI:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0, max_tokens=500)
    return _judge_llm


@dataclass
class HallucinationCheckResult:
    is_grounded: bool
    confidence: str  # "high" | "medium" | "low"
    unsupported_claims: list[str] = field(default_factory=list)
    check_failed: bool = False


def check_hallucination(answer: str, context_chunks: list[str]) -> HallucinationCheckResult:
    if not answer.strip():
        return HallucinationCheckResult(is_grounded=True, confidence="high", unsupported_claims=[])

    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no context retrieved)"

    messages = [
        SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=f"CONTEXT:\n{context_text}\n\nANSWER:\n{answer}"),
    ]

    try:
        response = _get_judge_llm().invoke(messages)
        parsed = _parse_judge_response(response.content)
    except Exception:
        return HallucinationCheckResult(
            is_grounded=True,  # fail open: don't block a response just because the judge call failed
            confidence="low",
            unsupported_claims=[],
            check_failed=True,
        )

    return HallucinationCheckResult(
        is_grounded=parsed.get("is_grounded", True),
        confidence=parsed.get("confidence", "low"),
        unsupported_claims=parsed.get("unsupported_claims", []),
    )


def _parse_judge_response(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"unsupported_claims": [], "is_grounded": True, "confidence": "low"}