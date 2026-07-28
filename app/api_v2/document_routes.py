"""
app/api_v2/document_routes.py

Document management endpoints for a session:
  GET    /api/v2/sessions/{session_id}/documents            -> list
  DELETE /api/v2/sessions/{session_id}/documents/{doc_id}    -> delete one
  DELETE /api/v2/sessions/{session_id}                       -> clear session
  POST   /api/v2/sessions/{session_id}/rebuild               -> rebuild index

Kept separate from upload_routes.py (upload-flow only) so each router stays
focused. Not yet included in main.py -- wired in the final step alongside
the other api_v2 routers.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.sessions import document_store, session_manager

router = APIRouter(prefix="/api/v2/sessions", tags=["documents"])


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    status: str
    uploaded_at: float
    chunk_count: int
    block_counts: dict
    warnings: list[str]
    error_message: str | None = None


class DocumentListResponse(BaseModel):
    session_id: str
    documents: list[DocumentSummary]


class DeleteResponse(BaseModel):
    doc_id: str
    chunks_removed: int


class ClearResponse(BaseModel):
    session_id: str
    cleared: bool


class RebuildResponse(BaseModel):
    session_id: str
    documents: list[DocumentSummary]


@router.get("/{session_id}/documents", response_model=DocumentListResponse)
def list_documents(session_id: str):
    _require_session(session_id)
    records = session_manager.list_documents(session_id)
    return DocumentListResponse(
        session_id=session_id,
        documents=[_to_summary(r) for r in records],
    )


@router.delete("/{session_id}/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(session_id: str, doc_id: str):
    _require_session(session_id)
    record = document_store.get_document(doc_id)
    if record is None or record.session_id != session_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    removed = session_manager.delete_document(session_id, doc_id)
    return DeleteResponse(doc_id=doc_id, chunks_removed=removed)


@router.delete("/{session_id}", response_model=ClearResponse)
def clear_session(session_id: str):
    _require_session(session_id)
    session_manager.clear_session(session_id)
    return ClearResponse(session_id=session_id, cleared=True)


@router.post("/{session_id}/rebuild", response_model=RebuildResponse)
def rebuild_index(session_id: str):
    _require_session(session_id)
    records = session_manager.rebuild_index(session_id)
    return RebuildResponse(
        session_id=session_id,
        documents=[_to_summary(r) for r in records],
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _require_session(session_id: str) -> None:
    if not document_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")


def _to_summary(record: document_store.DocumentRecord) -> DocumentSummary:
    return DocumentSummary(
        doc_id=record.doc_id,
        filename=record.filename,
        file_type=record.file_type,
        status=record.status,
        uploaded_at=record.uploaded_at,
        chunk_count=record.chunk_count,
        block_counts=record.block_counts,
        warnings=record.warnings,
        error_message=record.error_message,
    )