"""
app/sessions/document_store.py

SQLite-backed metadata store for sessions and documents.

This is intentionally separate from the vector backend (faiss_backend.py):
the vector backend only knows about embedded chunks, it has no concept of
"this upload is still indexing" or "this upload failed with these
warnings". This store owns that document-level bookkeeping, which the
upload progress bar and document-management endpoints (list/delete/rebuild,
file 14) both read from.

SQLite is used per the "zero setup" choice -- swapping to Postgres later
only means changing the connection string and DDL dialect in this one file;
nothing else in app/sessions or app/api_v2 talks to the database directly.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

DB_PATH = "session_documents.db"

_local = threading.local()
_init_lock = threading.Lock()
_initialized = False


@dataclass
class DocumentRecord:
    doc_id: str
    session_id: str
    filename: str
    file_type: str
    status: str  # "indexing" | "ready" | "failed"
    uploaded_at: float
    chunk_count: int = 0
    block_counts: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    error_message: str | None = None


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
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                last_active_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                status TEXT NOT NULL,
                uploaded_at REAL NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                block_counts TEXT NOT NULL DEFAULT '{}',
                warnings TEXT NOT NULL DEFAULT '[]',
                error_message TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_documents_session
                ON documents(session_id);
            """
        )
        conn.commit()
        _initialized = True


@contextmanager
def _cursor():
    conn = _get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------

def create_session(session_id: str | None = None) -> str:
    session_id = session_id or str(uuid.uuid4())
    now = time.time()
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO sessions (session_id, created_at, last_active_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET last_active_at = excluded.last_active_at
            """,
            (session_id, now, now),
        )
    return session_id


def touch_session(session_id: str) -> None:
    with _cursor() as cur:
        cur.execute(
            "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
            (time.time(), session_id),
        )


def session_exists(session_id: str) -> bool:
    with _cursor() as cur:
        cur.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
        return cur.fetchone() is not None


# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------

def register_document(
    session_id: str, filename: str, file_type: str, doc_id: str | None = None
) -> DocumentRecord:
    doc_id = doc_id or str(uuid.uuid4())
    record = DocumentRecord(
        doc_id=doc_id,
        session_id=session_id,
        filename=filename,
        file_type=file_type,
        status="indexing",
        uploaded_at=time.time(),
    )
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
                (doc_id, session_id, filename, file_type, status, uploaded_at,
                 chunk_count, block_counts, warnings, error_message)
            VALUES (?, ?, ?, ?, ?, ?, 0, '{}', '[]', NULL)
            """,
            (doc_id, session_id, filename, file_type, "indexing", record.uploaded_at),
        )
    return record


def mark_ready(doc_id: str, chunk_count: int, block_counts: dict, warnings: list) -> None:
    with _cursor() as cur:
        cur.execute(
            """
            UPDATE documents
            SET status = 'ready', chunk_count = ?, block_counts = ?, warnings = ?
            WHERE doc_id = ?
            """,
            (chunk_count, json.dumps(block_counts), json.dumps(warnings), doc_id),
        )


def mark_failed(doc_id: str, error_message: str) -> None:
    with _cursor() as cur:
        cur.execute(
            "UPDATE documents SET status = 'failed', error_message = ? WHERE doc_id = ?",
            (error_message, doc_id),
        )


def get_document(doc_id: str) -> DocumentRecord | None:
    with _cursor() as cur:
        cur.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = cur.fetchone()
    return _row_to_record(row) if row else None


def list_documents(session_id: str) -> list[DocumentRecord]:
    with _cursor() as cur:
        cur.execute(
            "SELECT * FROM documents WHERE session_id = ? ORDER BY uploaded_at DESC",
            (session_id,),
        )
        rows = cur.fetchall()
    return [_row_to_record(r) for r in rows]


def delete_document(doc_id: str) -> None:
    with _cursor() as cur:
        cur.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))


def clear_session_documents(session_id: str) -> None:
    with _cursor() as cur:
        cur.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))


def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row["doc_id"],
        session_id=row["session_id"],
        filename=row["filename"],
        file_type=row["file_type"],
        status=row["status"],
        uploaded_at=row["uploaded_at"],
        chunk_count=row["chunk_count"],
        block_counts=json.loads(row["block_counts"]),
        warnings=json.loads(row["warnings"]),
        error_message=row["error_message"],
    )