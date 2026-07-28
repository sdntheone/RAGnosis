"""
app/api_v2/upload_routes.py

Multi-file upload + automatic background indexing.

This router is NOT included in main.py's `app` yet -- that one-line wiring
step comes last (after all api_v2 routers exist), per the plan to keep
main.py's diff to a single additive line. Until then this file is inert
and imports cleanly on its own.

Flow:
  POST /api/v2/sessions                          -> create a session
  POST /api/v2/sessions/{session_id}/upload       -> upload N files, each
                                                      queued as a background
                                                      indexing job, returns
                                                      immediately with
                                                      status="indexing" per
                                                      file
  GET  /api/v2/sessions/{session_id}/documents/{doc_id}/status
                                                    -> poll one document's
                                                      indexing progress

Full document listing/delete/clear/rebuild live in document_routes.py
(kept separate: this file is upload-flow only).
"""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.sessions import document_store, session_manager

router = APIRouter(prefix="/api/v2/sessions", tags=["upload"])

SUPPORTED_EXTENSIONS = {
    "pdf", "docx", "pptx", "xlsx", "xlsm", "csv", "tsv",
    "txt", "md", "markdown", "png", "jpg", "jpeg", "webp", "bmp", "tiff",
}


class SessionCreateResponse(BaseModel):
    session_id: str


class UploadedFileStatus(BaseModel):
    doc_id: str
    filename: str
    status: str  # "indexing" | "ready" | "failed"


class UploadResponse(BaseModel):
    session_id: str
    files: list[UploadedFileStatus]


class DocumentStatusResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int
    block_counts: dict
    warnings: list[str]
    error_message: str | None = None


@router.post("", response_model=SessionCreateResponse)
def create_session():
    session_id = session_manager.get_or_create_session()
    return SessionCreateResponse(session_id=session_id)


@router.post("/{session_id}/upload", response_model=UploadResponse)
async def upload_documents(
    session_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    if not document_store.session_exists(session_id):
        session_manager.get_or_create_session(session_id)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    statuses: list[UploadedFileStatus] = []

    for upload in files:
        ext = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '.{ext}' for {upload.filename}. "
                       f"Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            )

        tmp_path = await _save_temp_file(upload)

        # Register immediately so the client can poll status right away,
        # even before the background task starts running.
        record = document_store.register_document(session_id, upload.filename, ext)

        background_tasks.add_task(
            _index_in_background, session_id, record.doc_id, tmp_path, upload.filename
        )

        statuses.append(
            UploadedFileStatus(doc_id=record.doc_id, filename=upload.filename, status="indexing")
        )

    return UploadResponse(session_id=session_id, files=statuses)


@router.get("/{session_id}/documents/{doc_id}/status", response_model=DocumentStatusResponse)
def get_document_status(session_id: str, doc_id: str):
    record = document_store.get_document(doc_id)
    if record is None or record.session_id != session_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    return DocumentStatusResponse(
        doc_id=record.doc_id,
        filename=record.filename,
        status=record.status,
        chunk_count=record.chunk_count,
        block_counts=record.block_counts,
        warnings=record.warnings,
        error_message=record.error_message,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

async def _save_temp_file(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await upload.read()
        tmp.write(content)
        return tmp.name


def _index_in_background(session_id: str, doc_id: str, tmp_path: str, filename: str) -> None:
    try:
        # process_upload registers its own record normally; here the record
        # already exists (created synchronously above so status polling
        # works immediately), so we reuse process_upload's internals via
        # its doc_id-aware path by re-registering is avoided -- instead we
        # directly call the same extract->chunk->tag->index steps against
        # the existing doc_id.
        session_manager.process_upload_existing_record(session_id, doc_id, tmp_path, filename)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)