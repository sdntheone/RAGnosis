"""
app/observability/tracing.py

Per-request tracing: times each pipeline stage and collects everything the
observability dashboard needs to show -- retrieval/embedding/generation
latency, token usage, similarity scores, retrieved chunks and sources,
guardrail status, and a confidence indicator.

Usage (see chat_stream_routes.py for the real wiring):

    trace = Trace(session_id=session_id, query=query)
    with trace.stage("embedding"):
        ...
    with trace.stage("retrieval"):
        chunks = retriever.get_relevant_documents(query)
    trace.record_retrieval(chunks, scores)
    with trace.stage("generation"):
        answer = llm.invoke(...)
    trace.record_generation(answer, token_usage)
    trace.record_guardrail("prompt_injection", passed=True)
    metrics_store.save_trace(trace)   # file 25

LangSmith integration: if LANGCHAIN_TRACING_V2=true is already set in the
environment (standard LangChain env var, not something this file invents),
every langchain_openai/langchain_core call in the existing pipeline is
already being traced to LangSmith automatically -- nothing extra needed
here. `Trace.langsmith_url` is a place for the caller to attach that run's
URL if it has one, so the dashboard can deep-link to it.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class RetrievedChunkInfo:
    content_preview: str  # first ~200 chars, not the full chunk (dashboard display only)
    source: str
    doc_id: str
    chunk_type: str
    similarity_score: float | None = None
    page_number: int | None = None


@dataclass
class GuardrailOutcome:
    name: str
    passed: bool
    detail: str | None = None


@dataclass
class Trace:
    session_id: str
    query: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)

    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    retrieved_chunks: list[RetrievedChunkInfo] = field(default_factory=list)
    guardrail_outcomes: list[GuardrailOutcome] = field(default_factory=list)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    answer: str | None = None
    groundedness_score: float | None = None
    confidence: str | None = None  # "high" | "medium" | "low" | "none"

    langsmith_url: str | None = None
    error: str | None = None

    _stage_start_times: dict[str, float] = field(default_factory=dict, repr=False)

    @contextmanager
    def stage(self, name: str):
        """Time a single pipeline stage. Nesting is fine (e.g. 'retrieval'
        containing 'dense_search' and 'bm25_search' as separate calls) --
        each name's latency is recorded independently and overwritten if
        the same stage name is used twice (callers should use distinct
        sub-stage names, e.g. 'retrieval.dense', 'retrieval.bm25', if they
        want both kept).
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stage_latencies_ms[name] = round(elapsed_ms, 2)

    def record_retrieval(self, chunks: list, scores: list[float] | None = None) -> None:
        scores = scores or [None] * len(chunks)
        for chunk, score in zip(chunks, scores):
            metadata = getattr(chunk, "metadata", {}) or {}
            content = getattr(chunk, "page_content", "") or ""
            self.retrieved_chunks.append(
                RetrievedChunkInfo(
                    content_preview=content[:200],
                    source=metadata.get("source", "unknown"),
                    doc_id=metadata.get("doc_id", "unknown"),
                    chunk_type=metadata.get("chunk_type", "text"),
                    similarity_score=score,
                    page_number=metadata.get("page_number"),
                )
            )

    def record_generation(self, answer: str, token_usage: dict | None = None) -> None:
        self.answer = answer
        if token_usage:
            self.prompt_tokens = token_usage.get("prompt_tokens")
            self.completion_tokens = token_usage.get("completion_tokens")
            self.total_tokens = token_usage.get("total_tokens")

    def record_guardrail(self, name: str, passed: bool, detail: str | None = None) -> None:
        self.guardrail_outcomes.append(GuardrailOutcome(name=name, passed=passed, detail=detail))

    def record_confidence(self, groundedness_score: float | None, confidence: str | None) -> None:
        self.groundedness_score = groundedness_score
        self.confidence = confidence

    def total_latency_ms(self) -> float:
        return round(sum(self.stage_latencies_ms.values()), 2)

    def all_guardrails_passed(self) -> bool:
        return all(g.passed for g in self.guardrail_outcomes)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "query": self.query,
            "started_at": self.started_at,
            "stage_latencies_ms": self.stage_latencies_ms,
            "total_latency_ms": self.total_latency_ms(),
            "retrieved_chunks": [
                {
                    "content_preview": c.content_preview,
                    "source": c.source,
                    "doc_id": c.doc_id,
                    "chunk_type": c.chunk_type,
                    "similarity_score": c.similarity_score,
                    "page_number": c.page_number,
                }
                for c in self.retrieved_chunks
            ],
            "guardrail_outcomes": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in self.guardrail_outcomes
            ],
            "all_guardrails_passed": self.all_guardrails_passed(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "answer": self.answer,
            "groundedness_score": self.groundedness_score,
            "confidence": self.confidence,
            "langsmith_url": self.langsmith_url,
            "error": self.error,
        }