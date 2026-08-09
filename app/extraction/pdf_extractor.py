"""
app/extraction/pdf_extractor.py

Multimodal PDF extraction: text, tables, images, OCR text inside images,
vision-LLM generated captions, figure/table captions, and reading-order
layout -- all linked together as Blocks (see app/extraction/base.py).

This file does NOT touch app/ingestion/pipeline.py. It is a new, optional
extraction path that ingestion/pipeline.py can start calling later if you
choose to -- today it continues to use PyMuPDFLoader exactly as before.

Libraries used:
- PyMuPDF (fitz)  -> text spans with font size (for heading detection),
                      embedded images, page geometry
- pdfplumber       -> table detection + extraction (PyMuPDF has no native
                      table support)
- pytesseract      -> OCR on extracted images (literal text inside images)
- Pillow           -> image I/O for OCR
- caption_generator -> vision-LLM captioning (app/extraction/caption_generator.py),
                       for describing image CONTENT (charts, diagrams, photos)
                       that OCR alone can't capture since it only reads
                       printed text, not visual meaning. Capped per document
                       via MAX_CAPTIONS_PER_DOCUMENT so a PDF with many images
                       doesn't fire an unbounded number of LLM calls at
                       upload time.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract

from app.extraction.base import (
    BaseExtractor,
    Block,
    BlockType,
    BoundingBox,
    ExtractionResult,
)

try:
    from app.extraction.caption_generator import generate_caption
except Exception:
    generate_caption = None  # caption_generator.py not present / import failed -- degrade gracefully

CAPTION_PREFIXES = ("figure", "fig.", "fig ", "table", "chart", "diagram")
HEADING_FONT_SIZE_RATIO = 1.25  # heading if span font size > 1.25x page median
MAX_CAPTIONS_PER_DOCUMENT = 15  # cap vision-LLM calls per uploaded PDF


class PdfExtractor(BaseExtractor):
    file_types = ("pdf",)

    def __init__(self, image_output_dir: str = "extracted_images"):
        self.image_output_dir = image_output_dir

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(self, file_path: str, doc_id: str) -> ExtractionResult:
        result = ExtractionResult(
            doc_id=doc_id,
            source_file=os.path.basename(file_path),
            file_type="pdf",
        )
        self._caption_count = 0  # reset per-document vision-LLM call counter

        doc_image_dir = os.path.join(self.image_output_dir, doc_id)
        os.makedirs(doc_image_dir, exist_ok=True)

        order_index = 0

        try:
            fitz_doc = fitz.open(file_path)
        except Exception as e:
            result.warnings.append(f"Failed to open PDF: {e}")
            return result

        try:
            plumber_doc = pdfplumber.open(file_path)
        except Exception as e:
            plumber_doc = None
            result.warnings.append(f"pdfplumber failed to open (tables skipped): {e}")

        for page_number in range(len(fitz_doc)):
            page = fitz_doc[page_number]

            text_blocks, order_index = self._extract_text_blocks(
                page, doc_id, result.source_file, page_number, order_index
            )
            result.blocks.extend(text_blocks)

            image_blocks, order_index = self._extract_images(
                fitz_doc, page, doc_id, result.source_file, page_number,
                order_index, doc_image_dir, result.warnings,
            )
            result.blocks.extend(image_blocks)

            if plumber_doc is not None:
                table_blocks, order_index = self._extract_tables(
                    plumber_doc, doc_id, result.source_file, page_number, order_index
                )
                result.blocks.extend(table_blocks)

        if plumber_doc is not None:
            plumber_doc.close()
        fitz_doc.close()

        self._link_captions_to_media(result)

        return result

    # ------------------------------------------------------------------
    # Text + heading detection
    # ------------------------------------------------------------------

    def _extract_text_blocks(
        self, page, doc_id, source_file, page_number, order_index
    ):
        blocks = []
        page_dict = page.get_text("dict")

        font_sizes = [
            span["size"]
            for block in page_dict.get("blocks", [])
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]
        median_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block, 1 = image block
                continue

            lines_text = []
            max_span_size = 0.0
            for line in block.get("lines", []):
                line_text = "".join(span["text"] for span in line.get("spans", []))
                lines_text.append(line_text)
                for span in line.get("spans", []):
                    max_span_size = max(max_span_size, span["size"])

            text = "\n".join(t for t in lines_text if t.strip())
            if not text.strip():
                continue

            is_heading = max_span_size >= median_size * HEADING_FONT_SIZE_RATIO
            is_caption = text.strip().lower().startswith(CAPTION_PREFIXES)

            block_type = (
                BlockType.CAPTION if is_caption
                else BlockType.HEADING if is_heading
                else BlockType.TEXT
            )

            bbox = block.get("bbox", (None, None, None, None))

            blocks.append(
                Block(
                    block_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    source_file=source_file,
                    block_type=block_type,
                    content=text,
                    page_number=page_number,
                    order_index=order_index,
                    bbox=BoundingBox(*bbox),
                )
            )
            order_index += 1

        return blocks, order_index

    # ------------------------------------------------------------------
    # Images + OCR + vision-LLM captioning
    # ------------------------------------------------------------------

    def _extract_images(
        self, fitz_doc, page, doc_id, source_file, page_number,
        order_index, doc_image_dir, warnings,
    ):
        blocks = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = fitz_doc.extract_image(xref)
                image_bytes = base_image["ext"]
                ext = base_image["ext"]
                image_filename = f"p{page_number}_img{img_index}.{ext}"
                image_path = os.path.join(doc_image_dir, image_filename)

                with open(image_path, "wb") as f:
                    f.write(base_image["image"])

                # Position on page (first occurrence rect, if available)
                rects = page.get_image_rects(xref)
                bbox = rects[0] if rects else fitz.Rect(0, 0, 0, 0)

                ocr_text = self._run_ocr(image_path, warnings)

                image_block = Block(
                    block_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    source_file=source_file,
                    block_type=BlockType.IMAGE,
                    content="",
                    image_path=image_path,
                    page_number=page_number,
                    order_index=order_index,
                    bbox=BoundingBox(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                )
                blocks.append(image_block)
                order_index += 1

                if ocr_text.strip():
                    ocr_block = Block(
                        block_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        source_file=source_file,
                        block_type=BlockType.OCR_TEXT,
                        content=ocr_text,
                        page_number=page_number,
                        order_index=order_index,
                        bbox=BoundingBox(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                        relates_to=[image_block.block_id],
                    )
                    image_block.relates_to.append(ocr_block.block_id)
                    blocks.append(ocr_block)
                    order_index += 1

                if generate_caption is not None and self._caption_count < MAX_CAPTIONS_PER_DOCUMENT:
                    try:
                        caption_text = generate_caption(image_path)
                    except Exception as e:
                        caption_text = ""
                        warnings.append(f"Caption generation failed: {e}")
                    self._caption_count += 1

                    if caption_text.strip():
                        caption_block = Block(
                            block_id=str(uuid.uuid4()),
                            doc_id=doc_id,
                            source_file=source_file,
                            block_type=BlockType.CAPTION,
                            content=caption_text,
                            page_number=page_number,
                            order_index=order_index,
                            bbox=BoundingBox(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                            relates_to=[image_block.block_id],
                        )
                        image_block.relates_to.append(caption_block.block_id)
                        blocks.append(caption_block)
                        order_index += 1

            except Exception as e:
                warnings.append(
                    f"Image extraction failed (page {page_number}, img {img_index}): {e}"
                )

        return blocks, order_index

    def _run_ocr(self, image_path: str, warnings: list) -> str:
        try:
            with Image.open(image_path) as im:
                return pytesseract.image_to_string(im)
        except Exception as e:
            warnings.append(f"OCR failed for {image_path}: {e}")
            return ""

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def _extract_tables(self, plumber_doc, doc_id, source_file, page_number, order_index):
        blocks = []
        try:
            plumber_page = plumber_doc.pages[page_number]
            tables = plumber_page.extract_tables()
        except Exception:
            return blocks, order_index

        for table in tables:
            if not table or not any(any(cell for cell in row) for row in table):
                continue

            markdown = self._table_to_markdown(table)

            blocks.append(
                Block(
                    block_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    source_file=source_file,
                    block_type=BlockType.TABLE,
                    content=markdown,
                    structured_data=table,
                    page_number=page_number,
                    order_index=order_index,
                )
            )
            order_index += 1

        return blocks, order_index

    @staticmethod
    def _table_to_markdown(rows: list[list[Optional[str]]]) -> str:
        clean_rows = [[cell or "" for cell in row] for row in rows]
        if not clean_rows:
            return ""
        header, *body = clean_rows
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Link captions to the nearest image/table on the same page
    # ------------------------------------------------------------------

    @staticmethod
    def _link_captions_to_media(result: ExtractionResult):
        by_page: dict[int, list[Block]] = {}
        for b in result.blocks:
            by_page.setdefault(b.page_number, []).append(b)

        for page_blocks in by_page.values():
            captions = [b for b in page_blocks if b.block_type == BlockType.CAPTION]
            media = [b for b in page_blocks if b.block_type in (BlockType.IMAGE, BlockType.TABLE)]

            for caption in captions:
                if not media:
                    continue
                cap_y = caption.bbox.y0 or 0
                nearest = min(
                    media,
                    key=lambda m: abs((m.bbox.y0 or 0) - cap_y),
                )
                result.link(caption, nearest)