"""
app/sessions/session_manager.py

Orchestrates one session's document lifecycle end to end:

    upload -> pick extractor by file type -> extract Blocks
            -> RelationalChunker -> Documents
            -> metadata_builder.build_metadata -> tagged Documents
            -> FaissSessionBackend.add_documents
            -> document_store record marked "ready"

Also owns:
  - one cached VectorStoreBackend instance per session_id (in-process cache;
    the actual data is persisted to disk by the backend itself, so this
    cache is just an optimization, not the source of truth)
  - a session-scoped BM25 retriever, rebuilt lazily from the backend's
    current documents whenever the session's chunk count changes
  - document management: list / delete / clear / rebuild

Raw uploaded files are kept under RAW_UPLOAD_ROOT/<session_id>/ so
rebuild_index() can re-run extraction without asking the user to re-upload.

Nothing here touches app/ingestion/pipeline.py or app/retrieval/*.
"""

from __future__ import annotations

import os
import shutil
import threading

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.extraction.base import ExtractionResult
from app.extraction.csv_extractor import CsvExtractor
from app.extraction.docx_extractor import DocxExtractor
from app.extraction.image_extractor import ImageExtractor
from app.extraction.pdf_extractor import PdfExtractor
from app.extraction.pptx_extractor import PptxExtractor
from app.extraction.xlsx_extractor import XlsxExtractor
from app.chunking.metadata_builder import DocumentContext, build_metadata
from app.chunking.relational_chunker import RelationalChunker
from app.sessions import document_store
from app.sessions.vector_backends.faiss_backend import FaissSessionBackend

RAW_UPLOAD_ROOT = os.path.join("uploads", "sessions")

TEXT_LIKE_TYPES = {"txt", "md", "markdown"}

_EXTRACTORS = [
    PdfExtractor(),
    DocxExtractor(),
    PptxExtractor(),
    XlsxExtractor(),
    CsvExtractor(),
    ImageExtractor(),
]

_backends: dict[str, FaissSessionBackend] = {}
_bm25_cache: dict[str, tuple[int, BM25Retriever | None]] = {}
_lock = threading.Lock()
_chunker = RelationalChunker()


# ----------------------------------------------------------------------
# Session + backend access
# ----------------------------------------------------------------------

def get_or_create_session(session_id: str | None = None) -> str:
    session_id = document_store.create_session(session_id)
    return session_id


def get_backend(session_id: str) -> FaissSessionBackend:
    with _lock:
        if session_id not in _backends:
            _backends[session_id] = FaissSessionBackend(session_id)
        return _backends[session_id]


def get_bm25_retriever(session_id: str, k: int = 10) -> BM25Retriever | None:
    backend = get_backend(session_id)
    current_count = backend.chunk_count()

    cached = _bm25_cache.get(session_id)
    if cached is not None and cached[0] == current_count:
        retriever = cached[1]
        if retriever is not None:
            retriever.k = k
        return retriever

    documents = backend.get_all_documents()
    retriever = BM25Retriever.from_documents(documents) if documents else None
    if retriever is not None:
        retriever.k = k

    _bm25_cache[session_id] = (current_count, retriever)
    return retriever


# ----------------------------------------------------------------------
# Upload -> extract -> chunk -> tag -> index
# ----------------------------------------------------------------------

def process_upload(session_id: str, file_path: str, filename: str) -> document_store.DocumentRecord:
    document_store.touch_session(session_id)
    file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    record = document_store.register_document(session_id, filename, file_type)
    return _index_document(session_id, record.doc_id, file_type, file_path, filename)


def process_upload_existing_record(
    session_id: str, doc_id: str, file_path: str, filename: str
) -> document_store.DocumentRecord:
    """Same as process_upload, but for a doc_id that was already registered
    synchronously (e.g. by upload_routes.py, so the client can poll status
    immediately, before the background indexing task has even started).
    """
    document_store.touch_session(session_id)
    file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _index_document(session_id, doc_id, file_type, file_path, filename)


def _index_document(
    session_id: str, doc_id: str, file_type: str, file_path: str, filename: str
) -> document_store.DocumentRecord:
    try:
        stored_path = _store_raw_upload(session_id, doc_id, file_path, filename)
        extraction = _extract(stored_path, file_type, doc_id, filename)

        chunks = _chunker.chunk(extraction.blocks)
        tagged_chunks = build_metadata(
            chunks,
            DocumentContext(
                doc_id=doc_id,
                session_id=session_id,
                source_file=filename,
                file_type=file_type,
            ),
        )

        backend = get_backend(session_id)
        backend.add_documents(tagged_chunks)

        document_store.mark_ready(
            doc_id,
            chunk_count=len(tagged_chunks),
            block_counts=extraction.block_count_by_type(),
            warnings=extraction.warnings,
        )

    except Exception as e:
        document_store.mark_failed(doc_id, str(e))
        raise

    return document_store.get_document(doc_id)

def _extract(file_path: str, file_type: str, doc_id: str, filename: str) -> ExtractionResult:
    if file_type in TEXT_LIKE_TYPES:
        return _extract_plain_text(file_path, file_type, doc_id, filename)

    for extractor in _EXTRACTORS:
        if extractor.supports(file_type):
            return extractor.extract(file_path, doc_id)

    raise ValueError(f"Unsupported file type: .{file_type}")


def _extract_plain_text(file_path: str, file_type: str, doc_id: str, filename: str) -> ExtractionResult:
    """TXT/Markdown have no tables/images/layout to speak of -- a single
    heading-free extractor inline here rather than a dedicated file.
    """
    from app.extraction.base import Block, BlockType

    result = ExtractionResult(doc_id=doc_id, source_file=filename, file_type=file_type)
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        result.warnings.append(f"Failed to read text file: {e}")
        return result

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for i, para in enumerate(paragraphs):
        result.blocks.append(
            Block(
                block_id=f"{doc_id}-p{i}",
                doc_id=doc_id,
                source_file=filename,
                block_type=BlockType.TEXT,
                content=para,
                order_index=i,
            )
        )
    return result


def _store_raw_upload(session_id: str, doc_id: str, file_path: str, filename: str) -> str:
    session_dir = os.path.join(RAW_UPLOAD_ROOT, session_id)
    os.makedirs(session_dir, exist_ok=True)
    stored_path = os.path.join(session_dir, f"{doc_id}_{filename}")
    if os.path.abspath(stored_path) != os.path.abspath(file_path):
        shutil.copyfile(file_path, stored_path)
    return stored_path


# ----------------------------------------------------------------------
# Document management
# ----------------------------------------------------------------------

def list_documents(session_id: str) -> list[document_store.DocumentRecord]:
    return document_store.list_documents(session_id)


def delete_document(session_id: str, doc_id: str) -> int:
    backend = get_backend(session_id)
    removed = backend.delete_document(doc_id)
    document_store.delete_document(doc_id)

    session_dir = os.path.join(RAW_UPLOAD_ROOT, session_id)
    if os.path.isdir(session_dir):
        for fname in os.listdir(session_dir):
            if fname.startswith(f"{doc_id}_"):
                os.remove(os.path.join(session_dir, fname))

    return removed


def clear_session(session_id: str) -> None:
    backend = get_backend(session_id)
    backend.clear()
    document_store.clear_session_documents(session_id)
    _bm25_cache.pop(session_id, None)

    session_dir = os.path.join(RAW_UPLOAD_ROOT, session_id)
    if os.path.isdir(session_dir):
        shutil.rmtree(session_dir, ignore_errors=True)


def rebuild_index(session_id: str) -> list[document_store.DocumentRecord]:
    """Re-run extraction + chunking for every raw file still stored for
    this session, without requiring re-upload. Useful after changing
    chunking parameters or recovering from a partial failure.
    """
    backend = get_backend(session_id)
    backend.clear()
    _bm25_cache.pop(session_id, None)

    existing_records = {r.doc_id: r for r in document_store.list_documents(session_id)}
    document_store.clear_session_documents(session_id)

    session_dir = os.path.join(RAW_UPLOAD_ROOT, session_id)
    results = []
    if not os.path.isdir(session_dir):
        return results

    for fname in os.listdir(session_dir):
        doc_id, _, original_name = fname.partition("_")
        old_record = existing_records.get(doc_id)
        display_name = old_record.filename if old_record else original_name
        full_path = os.path.join(session_dir, fname)
        try:
            record = process_upload(session_id, full_path, display_name)
            results.append(record)
        except Exception:
            continue  # already marked failed inside process_upload

    return results