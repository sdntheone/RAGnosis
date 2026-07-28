"""
app/api_v2/chat_stream_routes.py

Streaming chat endpoint for a session, wiring together:
  - session-scoped hybrid retrieval (dense FAISS + sparse BM25 + RRF fusion
    + cross-encoder rerank) -- reuses app/retrieval/rrf.py and
    app/retrieval/reranker.py UNCHANGED; only the retrievers feeding them
    are session-scoped instead of the global ones in app/retrieval/*
  - metadata filtering (app/chunking/metadata_builder.build_filter)
  - the full guardrail pipeline (app/guardrails/*)
  - streaming generation (Server-Sent Events)
  - trace recording (app/observability/*)

Prompt templates are reused as-is from app/llm/prompt.py (get_prompt) --
no changes there either.

Endpoint:
  POST /api/v2/sessions/{session_id}/chat/stream

Known shortcut (see chat message above the code block): scored dense
search reaches into FaissSessionBackend._store directly, since
VectorStoreBackend's interface (file 11) only exposes unscored
similarity_search(). See _dense_search_with_scores().
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.llm.prompt import get_prompt
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker
from app.chunking.metadata_builder import build_filter
from app.sessions import document_store, session_manager
from app.sessions.vector_backends.faiss_backend import FaissSessionBackend, _matches_filter
from app.guardrails import (
    input_validation,
    prompt_injection,
    jailbreak_detection,
    retrieval_validation,
    groundedness,
    hallucination_check,
    pii_masking,
)
from app.observability import metrics_store
from app.observability.tracing import Trace

from langchain_openai import ChatOpenAI

router = APIRouter(prefix="/api/v2/sessions", tags=["chat"])

RERANK_TOP_K = 5
DENSE_FETCH_K = 15
SPARSE_FETCH_K = 15
GROUNDEDNESS_JUDGE_THRESHOLD = 0.35  # below this, pay for the LLM-as-judge check

_reranker: CrossEncoderReranker | None = None


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


class ChatStreamRequest(BaseModel):
    query: str
    mode: str = "default"
    k: int = RERANK_TOP_K
    file_types: list[str] | None = None
    chunk_types: list[str] | None = None
    has_media: bool | None = None
    doc_ids: list[str] | None = None


@router.post("/{session_id}/chat/stream")
async def chat_stream(session_id: str, request: ChatStreamRequest):
    if not document_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    return StreamingResponse(
        _stream_response(session_id, request),
        media_type="text/event-stream",
    )


# ------------------------------------------------------------------
# Main generator: guardrails -> retrieval -> generation -> post-checks
# ------------------------------------------------------------------

def _stream_response(session_id: str, request: ChatStreamRequest):
    trace = Trace(session_id=session_id, query=request.query)

    try:
        # --- 1. Input validation ---------------------------------------
        validation = input_validation.validate_query(request.query)
        trace.record_guardrail("input_validation", validation.is_valid, validation.reason)
        if not validation.is_valid:
            yield from _emit_blocked(trace, f"Invalid input: {validation.reason}")
            return
        query = validation.cleaned_query

        # --- 2. Jailbreak + prompt-injection check on the query ---------
        jailbreak_result = jailbreak_detection.detect(query)
        trace.record_guardrail(
            "jailbreak_detection", not jailbreak_result.is_suspicious,
            f"matched: {jailbreak_result.matched_patterns}" if jailbreak_result.is_suspicious else None,
        )
        if jailbreak_result.is_suspicious:
            yield from _emit_blocked(
                trace, "This request looks like an attempt to bypass the assistant's normal behavior."
            )
            return

        injection_result = prompt_injection.detect_in_query(query)
        trace.record_guardrail(
            "prompt_injection_query", not injection_result.is_suspicious,
            f"matched: {injection_result.matched_patterns}" if injection_result.is_suspicious else None,
        )
        if injection_result.is_suspicious:
            yield from _emit_blocked(
                trace, "This request looks like an attempt to override the assistant's instructions."
            )
            return

        # --- 3. Retrieval (session-scoped hybrid: dense + sparse + RRF + rerank) ---
        metadata_filter = build_filter(
            file_types=request.file_types,
            chunk_types=request.chunk_types,
            has_media=request.has_media,
            doc_ids=request.doc_ids,
        )

        with trace.stage("retrieval.dense"):
            dense_docs, dense_scores = _dense_search_with_scores(
                session_id, query, k=DENSE_FETCH_K, metadata_filter=metadata_filter
            )

        with trace.stage("retrieval.sparse"):
            bm25 = session_manager.get_bm25_retriever(session_id, k=SPARSE_FETCH_K)
            sparse_docs = bm25.get_relevant_documents(query) if bm25 else []
            if metadata_filter:
                sparse_docs = [d for d in sparse_docs if _matches_filter(d.metadata, metadata_filter)]

        with trace.stage("retrieval.fusion"):
            fused_docs = reciprocal_rank_fusion(dense_docs, sparse_docs)

        with trace.stage("retrieval.rerank"):
            reranked_docs = _get_reranker().rerank(query, fused_docs, top_k=request.k)

        # --- 4. Prompt-injection check on retrieved CONTEXT --------------
        chunk_texts = [d.page_content for d in reranked_docs]
        context_injection_results = prompt_injection.detect_in_context(chunk_texts)
        trace.record_guardrail(
            "prompt_injection_context",
            not context_injection_results,
            f"{len(context_injection_results)} chunk(s) flagged" if context_injection_results else None,
        )
        if context_injection_results:
            flagged_texts = set()
            for text, result in zip(chunk_texts, [
                prompt_injection.detect_in_context([t]) for t in chunk_texts
            ]):
                if result:
                    flagged_texts.add(text)
            reranked_docs = [d for d in reranked_docs if d.page_content not in flagged_texts]

        # --- 5. Retrieval validation (similarity floor) -------------------
        score_by_content = dict(zip([d.page_content for d in dense_docs], dense_scores))
        reranked_scores = [score_by_content.get(d.page_content, 0.0) for d in reranked_docs]

        retrieval_check = retrieval_validation.validate_retrieval(reranked_scores)
        trace.record_guardrail(
            "retrieval_validation", retrieval_check.is_valid, retrieval_check.reason
        )
        trace.record_retrieval(reranked_docs, reranked_scores)

        if not retrieval_check.is_valid:
            yield from _emit_blocked(
                trace,
                "I don't have enough relevant information in the uploaded documents to answer that.",
                is_no_info=True,
            )
            return

        # --- 6. Generation (streaming) ------------------------------------
        prompt_template = get_prompt(request.mode)
        context_text = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}"
            f"{', page ' + str(d.metadata['page_number']) if d.metadata.get('page_number') is not None else ''}]\n"
            f"{d.page_content}"
            for d in reranked_docs
        )
        messages = prompt_template.format_messages(context=context_text, question=query)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

        full_answer = ""
        with trace.stage("generation"):
            for chunk in llm.stream(messages):
                token = chunk.content or ""
                full_answer += token
                if token:
                    yield _sse_event("token", {"text": token})

        trace.record_generation(full_answer)

        # --- 7. Groundedness + selective hallucination check --------------
        with trace.stage("groundedness_check"):
            groundedness_result = groundedness.score_groundedness(full_answer, chunk_texts)

        confidence = retrieval_check.confidence
        if groundedness_result.score < GROUNDEDNESS_JUDGE_THRESHOLD:
            with trace.stage("hallucination_check"):
                hallucination_result = hallucination_check.check_hallucination(full_answer, chunk_texts)
            trace.record_guardrail(
                "hallucination_check", hallucination_result.is_grounded,
                f"unsupported claims: {hallucination_result.unsupported_claims}"
                if not hallucination_result.is_grounded else None,
            )
            if not hallucination_result.is_grounded:
                confidence = "low"

        trace.record_confidence(groundedness_result.score, confidence)

        # --- 8. Sources / citations + confidence, sent as final event -----
        sources = [
            {
                "source": d.metadata.get("source"),
                "page_number": d.metadata.get("page_number"),
                "chunk_type": d.metadata.get("chunk_type"),
                "doc_id": d.metadata.get("doc_id"),
            }
            for d in reranked_docs
        ]
        yield _sse_event("done", {
            "sources": sources,
            "confidence": confidence,
            "groundedness_score": groundedness_result.score,
        })

    except Exception as e:
        trace.error = str(e)
        yield _sse_event("error", {"message": "Something went wrong generating a response."})

    finally:
        _mask_trace_for_logging(trace)
        metrics_store.save_trace(trace)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _dense_search_with_scores(
    session_id: str, query: str, k: int, metadata_filter: dict | None
) -> tuple[list, list[float]]:
    """Reaches into FaissSessionBackend's internal FAISS store to get
    similarity scores, since VectorStoreBackend.similarity_search() (file
    11) only returns unscored Documents. See the note at the top of this
    file -- this is a documented shortcut, not the intended long-term shape
    of the abstraction.
    """
    backend = session_manager.get_backend(session_id)
    assert isinstance(backend, FaissSessionBackend)

    if backend._store is None:
        return [], []

    fetch_k = k * 5 if metadata_filter else k
    results = backend._store.similarity_search_with_score(query, k=fetch_k)

    docs, distances = [], []
    for doc, distance in results:
        if metadata_filter and not _matches_filter(doc.metadata, metadata_filter):
            continue
        docs.append(doc)
        distances.append(retrieval_validation.faiss_distance_to_similarity(distance))

    return docs[:k], distances[:k]


def _emit_blocked(trace: Trace, message: str, is_no_info: bool = False):
    trace.answer = message
    yield _sse_event("blocked" if not is_no_info else "no_info", {"message": message})


def _sse_event(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


def _mask_trace_for_logging(trace: Trace) -> None:
    """Mask PII in the query/answer before they're persisted to the trace
    store -- the response already streamed to the client unmasked; this
    only affects what lands in observability logs.
    """
    trace.query = pii_masking.mask(trace.query).masked_text
    if trace.answer:
        trace.answer = pii_masking.mask(trace.answer).masked_text