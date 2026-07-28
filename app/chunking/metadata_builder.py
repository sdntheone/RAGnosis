"""
app/chunking/metadata_builder.py

Enriches chunk metadata after relational_chunker.py produces the initial
Document objects. relational_chunker.py already attaches content-level
metadata (chunk_type, block_ids, page_numbers, image_path); this module
adds document- and session-level metadata that the chunker has no way to
know about on its own (which session/user this upload belongs to, when it
was uploaded, the original filename and file type, a stable doc_id).

Kept as a separate step (rather than folded into the chunker) so the
chunker stays a pure content->chunks transform, and this stays a pure
metadata-tagging transform -- easier to test and reuse independently
(e.g. metadata_builder can also be used to retag existing chunks if a
document gets renamed or moved to a different session).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class DocumentContext:
    """Everything known about the source document at upload time, needed
    to tag its chunks for later filtering during retrieval.
    """

    doc_id: str
    session_id: str
    source_file: str
    file_type: str  # "pdf" | "docx" | "pptx" | "xlsx" | "csv" | "txt" | "md" | "image"
    uploaded_at: float | None = None  # unix timestamp; defaults to now if unset
    uploaded_by: str | None = None
    extra_tags: dict | None = None


def build_metadata(chunks: list[Document], context: DocumentContext) -> list[Document]:
    """Return new Document objects with document/session-level metadata
    merged in. Does not mutate the input list.

    Content-level keys already set by relational_chunker.py (chunk_type,
    block_ids, block_types, page_numbers, sibling_block_ids, image_path)
    are preserved as-is; this function only adds keys the chunker doesn't
    have visibility into.
    """
    uploaded_at = context.uploaded_at if context.uploaded_at is not None else time.time()

    doc_level_metadata = {
        "doc_id": context.doc_id,
        "session_id": context.session_id,
        "source": context.source_file,
        "file_type": context.file_type,
        "uploaded_at": uploaded_at,
        "uploaded_by": context.uploaded_by,
        **(context.extra_tags or {}),
    }

    enriched: list[Document] = []
    for chunk in chunks:
        merged_metadata = {**chunk.metadata, **doc_level_metadata}
        merged_metadata["chunk_char_count"] = len(chunk.page_content)
        merged_metadata["has_media"] = merged_metadata.get("chunk_type") in (
            "table", "image", "ocr_text", "image_caption_only",
        )
        enriched.append(Document(page_content=chunk.page_content, metadata=merged_metadata))

    return enriched


def build_filter(
    session_id: str | None = None,
    file_types: list[str] | None = None,
    chunk_types: list[str] | None = None,
    has_media: bool | None = None,
    doc_ids: list[str] | None = None,
) -> dict:
    """Build a metadata filter dict for use at retrieval time.

    Kept as a plain dict (not backend-specific) so both the FAISS backend
    (file 11) and a future non-FAISS backend can each translate it into
    whatever native filter syntax they need, without this module knowing
    which backend is active.
    """
    filt: dict = {}
    if session_id is not None:
        filt["session_id"] = session_id
    if file_types:
        filt["file_type"] = {"$in": file_types}
    if chunk_types:
        filt["chunk_type"] = {"$in": chunk_types}
    if has_media is not None:
        filt["has_media"] = has_media
    if doc_ids:
        filt["doc_id"] = {"$in": doc_ids}
    return filt