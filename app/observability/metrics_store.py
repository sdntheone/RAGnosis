"""
app/observability/metrics_store.py

Persists Trace objects (app/observability/tracing.py) to SQLite and
provides aggregation queries for the observability dashboard: recent
traces, average per-stage latency, guardrail pass rates, token usage over
time, similarity score distribution.

Separate SQLite file from app/sessions/document_store.py -- traces are a
much higher-write-volume, append-mostly table (one row per chat query) and
keeping it separate avoids lock contention with document upload/management
operations.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time

from app.observability.tracing import Trace

DB_PATH = "observability_traces.db"

_local = threading.local()
_init_lock = threading.Lock()
_initialized = False


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.execute("PRAGMA journal_mode=WAL;")
        _local.conn.row_factory = sqlite3.Row
    _ensure_schema(_local.conn)
    return _local.conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                started_at REAL NOT NULL,
                total_latency_ms REAL NOT NULL,
                stage_latencies_ms TEXT NOT NULL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                top_similarity_score REAL,
                confidence TEXT,
                groundedness_score REAL,
                all_guardrails_passed INTEGER NOT NULL,
                error TEXT,
                payload TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_traces_session
                ON traces(session_id);
            CREATE INDEX IF NOT EXISTS idx_traces_started_at
                ON traces(started_at);
            """
        )
        conn.commit()
        _initialized = True


def save_trace(trace: Trace) -> None:
    payload = trace.to_dict()
    scores = [
        float(c.similarity_score) for c in trace.retrieved_chunks if c.similarity_score is not None
    ]
    top_score = max(scores) if scores else None

    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO traces
            (trace_id, session_id, started_at, total_latency_ms, stage_latencies_ms,
             prompt_tokens, completion_tokens, total_tokens, top_similarity_score,
             confidence, groundedness_score, all_guardrails_passed, error, payload)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trace.trace_id,
            trace.session_id,
            trace.started_at,
            trace.total_latency_ms(),
            json.dumps(trace.stage_latencies_ms),
            trace.prompt_tokens,
            trace.completion_tokens,
            trace.total_tokens,
            top_score,
            trace.confidence,
            trace.groundedness_score,
            int(trace.all_guardrails_passed()),
            trace.error,
            json.dumps(payload),
        ),
    )
    conn.commit()


def get_recent_traces(session_id: str | None = None, limit: int = 50) -> list[dict]:
    conn = _get_conn()
    if session_id:
        cur = conn.execute(
            "SELECT payload FROM traces WHERE session_id = ? ORDER BY started_at DESC LIMIT ?",
            (session_id, limit),
        )
    else:
        cur = conn.execute(
            "SELECT payload FROM traces ORDER BY started_at DESC LIMIT ?", (limit,)
        )
    return [json.loads(row["payload"]) for row in cur.fetchall()]


def get_summary_stats(session_id: str | None = None, since_seconds: float | None = None) -> dict:
    """Aggregate stats for the dashboard header: avg latency per stage,
    avg tokens, guardrail pass rate, avg similarity score.
    """
    conn = _get_conn()

    where_clauses = []
    params: list = []
    if session_id:
        where_clauses.append("session_id = ?")
        params.append(session_id)
    if since_seconds is not None:
        where_clauses.append("started_at >= ?")
        params.append(time.time() - since_seconds)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    cur = conn.execute(
        f"""
        SELECT
            COUNT(*) AS request_count,
            AVG(total_latency_ms) AS avg_total_latency_ms,
            AVG(prompt_tokens) AS avg_prompt_tokens,
            AVG(completion_tokens) AS avg_completion_tokens,
            AVG(total_tokens) AS avg_total_tokens,
            AVG(top_similarity_score) AS avg_similarity_score,
            AVG(groundedness_score) AS avg_groundedness_score,
            SUM(all_guardrails_passed) AS guardrail_pass_count,
            SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS error_count
        FROM traces
        {where_sql}
        """,
        params,
    )
    row = cur.fetchone()

    request_count = row["request_count"] or 0
    guardrail_pass_rate = (
        row["guardrail_pass_count"] / request_count if request_count else None
    )

    # Per-stage average latency requires unpacking the JSON column --
    # done in Python since stage names are dynamic (not fixed columns).
    stage_totals: dict[str, list[float]] = {}
    cur = conn.execute(f"SELECT stage_latencies_ms FROM traces {where_sql}", params)
    for r in cur.fetchall():
        stages = json.loads(r["stage_latencies_ms"])
        for stage_name, latency in stages.items():
            stage_totals.setdefault(stage_name, []).append(latency)

    avg_stage_latencies_ms = {
        stage: round(sum(values) / len(values), 2) for stage, values in stage_totals.items()
    }

    return {
        "request_count": request_count,
        "avg_total_latency_ms": _round_or_none(row["avg_total_latency_ms"]),
        "avg_stage_latencies_ms": avg_stage_latencies_ms,
        "avg_prompt_tokens": _round_or_none(row["avg_prompt_tokens"]),
        "avg_completion_tokens": _round_or_none(row["avg_completion_tokens"]),
        "avg_total_tokens": _round_or_none(row["avg_total_tokens"]),
        "avg_similarity_score": _round_or_none(row["avg_similarity_score"], 4),
        "avg_groundedness_score": _round_or_none(row["avg_groundedness_score"], 4),
        "guardrail_pass_rate": _round_or_none(guardrail_pass_rate, 4),
        "error_count": row["error_count"] or 0,
    }


def _round_or_none(value, digits: int = 2):
    return round(value, digits) if value is not None else None