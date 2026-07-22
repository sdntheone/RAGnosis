"""
app/extraction/base.py

Shared data model for the multimodal extraction layer.

This module is NEW and is not imported anywhere in the existing pipeline
(app/ingestion/pipeline.py, app/retrieval/*, app/llm/*). It introduces the
common schema that every format-specific extractor (PDF, DOCX, PPTX, XLSX,
CSV, image) will produce, and that the relationship-aware chunker and
metadata builder will consume downstream.

Design goals
------------
1. Preserve *layout position* (page, order, bounding box) so blocks can be
   reassembled in reading order.
2. Preserve *relationships* between a piece of text, the table it refers to,
   an image, and its caption -- so a retrieved chunk can carry its neighbors
   along with it instead of losing that context.
3. Stay LangChain-compatible: `Block.to_langchain_document()` produces a
   `langchain_core.documents.Document`, so everything downstream (FAISS,
   BM25, the existing HybridRetriever) can consume these blocks without any
   change to app/retrieval/*.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from langchain_core.documents import Document


class BlockType(str, Enum):
    """The kind of content a Block holds."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    OCR_TEXT = "ocr_text"
    HEADING = "heading"


@dataclass
class BoundingBox:
    """Position of a block on a page, in PDF/points or pixel space.

    All values are optional because not every source format has real
    coordinates (e.g. a DOCX paragraph does not have an (x, y) position the
    way a PDF text span does).
    """

    x0: Optional[float] = None
    y0: Optional[float] = None
    x1: Optional[float] = None
    y1: Optional[float] = None

    def as_tuple(self) -> Optional[tuple]:
        if None in (self.x0, self.y0, self.x1, self.y1):
            return None
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class Block:
    """A single extracted unit of content: a paragraph, a table, an image,
    a caption, or OCR'd text found inside an image.

    `relates_to` holds the ids of other Blocks this one is linked to (e.g. a
    caption's `relates_to` includes the image id it describes; a table's
    `relates_to` includes any caption/heading directly above it). This is
    what the relational chunker uses to keep text + table + image + caption
    together instead of splitting them into unrelated chunks.
    """

    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    source_file: str = ""
    block_type: BlockType = BlockType.TEXT

    # The actual content. For TEXT/CAPTION/HEADING/OCR_TEXT this is a string.
    # For TABLE this is a markdown-rendered table (so it stays embeddable as
    # text) with the raw structured rows kept in `structured_data`.
    # For IMAGE this is empty; the image bytes live on disk, referenced by
    # `image_path`, and any generated caption is a *separate* CAPTION block
    # linked via `relates_to`.
    content: str = ""

    structured_data: Optional[list[list[str]]] = None  # raw table rows, if TABLE
    image_path: Optional[str] = None  # saved image file, if IMAGE

    page_number: Optional[int] = None
    order_index: int = 0  # reading-order position within the document
    bbox: BoundingBox = field(default_factory=BoundingBox)

    relates_to: list[str] = field(default_factory=list)  # other block_ids

    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        basis = f"{self.doc_id}:{self.block_type}:{self.content[:500]}"
        return hashlib.sha256(basis.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def to_langchain_document(self) -> Document:
        """Convert to a LangChain Document so it can flow through the
        existing FAISS / BM25 / HybridRetriever code unmodified.
        """
        metadata = {
            "block_id": self.block_id,
            "doc_id": self.doc_id,
            "source": self.source_file,
            "block_type": self.block_type.value,
            "page_number": self.page_number,
            "order_index": self.order_index,
            "relates_to": list(self.relates_to),
            "bbox": self.bbox.as_tuple(),
            "image_path": self.image_path,
            **self.extra_metadata,
        }
        return Document(page_content=self.content, metadata=metadata)


@dataclass
class ExtractionResult:
    """Everything extracted from one uploaded file."""

    doc_id: str
    source_file: str
    file_type: str  # "pdf" | "docx" | "pptx" | "xlsx" | "csv" | "txt" | "md" | "image"
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def block_count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for b in self.blocks:
            counts[b.block_type.value] = counts.get(b.block_type.value, 0) + 1
        return counts

    def link(self, a: Block, b: Block) -> None:
        """Create a bidirectional relationship between two blocks."""
        if b.block_id not in a.relates_to:
            a.relates_to.append(b.block_id)
        if a.block_id not in b.relates_to:
            b.relates_to.append(a.block_id)


class BaseExtractor:
    """Interface every format-specific extractor implements.

    Kept deliberately tiny: one method in, one ExtractionResult out. Each
    concrete extractor (PdfExtractor, DocxExtractor, ...) owns its own
    library choices (pymupdf, pdfplumber, python-docx, pytesseract, etc.)
    internally -- callers never need to know which library was used.
    """

    file_types: tuple[str, ...] = ()

    def extract(self, file_path: str, doc_id: str) -> ExtractionResult:
        raise NotImplementedError

    def supports(self, file_type: str) -> bool:
        return file_type.lower().lstrip(".") in self.file_types