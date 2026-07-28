"""
app/api_v2/observability_routes.py

Read-only endpoints over app/observability/metrics_store.py, feeding the
dashboard tab in streamlit_app_v2.py:

  GET /api/v2/sessions/{session_id}/observability/summary
      -> aggregate stats: avg latency per stage, avg tokens, guardrail
         pass rate, avg similarity/groundedness scores

  GET /api/v2/sessions/{session_id}/observability/traces
      -> recent individual traces for this session (retrieved chunks,
         sources, guardrail outcomes, per-stage latency)

Not yet included in main.py -- wired in the final step alongside the other
api_v2 routers.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.sessions import document_store
from app.observability import metrics_store

router = APIRouter(prefix="/api/v2/sessions", tags=["observability"])


@router.get("/{session_id}/observability/summary")
def get_summary(session_id: str, since_seconds: float | None = Query(default=None)):
    _require_session(session_id)
    return metrics_store.get_summary_stats(session_id=session_id, since_seconds=since_seconds)


@router.get("/{session_id}/observability/traces")
def get_traces(session_id: str, limit: int = Query(default=50, le=200)):
    _require_session(session_id)
    traces = metrics_store.get_recent_traces(session_id=session_id, limit=limit)
    return {"session_id": session_id, "traces": traces}


def _require_session(session_id: str) -> None:
    if not document_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")