"""
app/extraction/docx_extractor.py

Multimodal DOCX extraction: paragraphs (with heading/caption detection),
tables, embedded images, OCR text inside images, vision-LLM generated
captions, and document-authored captions -- linked together as Blocks
(see app/extraction/base.py), preserving document reading order.

Does not touch app/ingestion/pipeline.py.
"""

from __future__ import annotations

import os
import uuid

from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image
import pytesseract

from app.extraction.base import (
    BaseExtractor,
    Block,
    BlockType,
    ExtractionResult,
)

try:
    from app.extraction.caption_generator import generate_caption
except Exception:
    generate_caption = None  # caption_generator.py not present / import failed -- degrade gracefully

CAPTION_PREFIXES = ("figure", "fig.", "fig ", "table", "chart", "diagram")
CAPTION_LINK_WINDOW = 2  # order_index distance within which a caption links to media
MAX_CAPTIONS_PER_DOCUMENT = 15  # cap vision-LLM calls per uploaded DOCX


class DocxExtractor(BaseExtractor):
    file_types = ("docx",)

    def __init__(self, image_output_dir: str = "extracted_images"):
        self.image_output_dir = image_output_dir

    def extract(self, file_path: str, doc_id: str) -> ExtractionResult:
        result = ExtractionResult(
            doc_id=doc_id,
            source_file=os.path.basename(file_path),
            file_type="docx",
        )
        self._caption_count = 0  # reset per-document vision-LLM call counter

        doc_image_dir = os.path.join(self.image_output_dir, doc_id)
        os.makedirs(doc_image_dir, exist_ok=True)

        try:
            docx_doc = DocxDocument(file_path)
        except Exception as e:
            result.warnings.append(f"Failed to open DOCX: {e}")
            return result

        order_index = 0
        image_counter = 0

        for item in self._iter_block_items(docx_doc):
            if isinstance(item, Paragraph):
                text = item.text.strip()

                # Embedded images inside this paragraph's runs
                for rid in self._image_rids_in_paragraph(item):
                    try:
                        image_part = docx_doc.part.related_parts[rid]
                        image_counter += 1
                        ext = image_part.content_type.split("/")[-1]
                        image_filename = f"img{image_counter}.{ext}"
                        image_path = os.path.join(doc_image_dir, image_filename)
                        with open(image_path, "wb") as f:
                            f.write(image_part.blob)

                        ocr_text = self._run_ocr(image_path, result.warnings)

                        image_block = Block(
                            block_id=str(uuid.uuid4()),
                            doc_id=doc_id,
                            source_file=result.source_file,
                            block_type=BlockType.IMAGE,
                            content="",
                            image_path=image_path,
                            order_index=order_index,
                        )
                        result.blocks.append(image_block)
                        order_index += 1

                        if ocr_text.strip():
                            ocr_block = Block(
                                block_id=str(uuid.uuid4()),
                                doc_id=doc_id,
                                source_file=result.source_file,
                                block_type=BlockType.OCR_TEXT,
                                content=ocr_text,
                                order_index=order_index,
                                relates_to=[image_block.block_id],
                            )
                            image_block.relates_to.append(ocr_block.block_id)
                            result.blocks.append(ocr_block)
                            order_index += 1

                        if generate_caption is not None and self._caption_count < MAX_CAPTIONS_PER_DOCUMENT:
                            try:
                                caption_text = generate_caption(image_path)
                            except Exception as e:
                                caption_text = ""
                                result.warnings.append(f"Caption generation failed: {e}")
                            self._caption_count += 1

                            if caption_text.strip():
                                caption_block = Block(
                                    block_id=str(uuid.uuid4()),
                                    doc_id=doc_id,
                                    source_file=result.source_file,
                                    block_type=BlockType.CAPTION,
                                    content=caption_text,
                                    order_index=order_index,
                                    relates_to=[image_block.block_id],
                                )
                                image_block.relates_to.append(caption_block.block_id)
                                result.blocks.append(caption_block)
                                order_index += 1

                    except Exception as e:
                        result.warnings.append(f"Image extraction failed: {e}")

                if not text:
                    continue

                style_name = (item.style.name or "").lower()
                is_heading = style_name.startswith("heading") or style_name == "title"
                is_caption = style_name == "caption" or text.lower().startswith(CAPTION_PREFIXES)

                block_type = (
                    BlockType.CAPTION if is_caption
                    else BlockType.HEADING if is_heading
                    else BlockType.TEXT
                )

                result.blocks.append(
                    Block(
                        block_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        source_file=result.source_file,
                        block_type=block_type,
                        content=text,
                        order_index=order_index,
                    )
                )
                order_index += 1

            elif isinstance(item, Table):
                markdown, rows = self._table_to_markdown(item)
                if not markdown.strip():
                    continue
                result.blocks.append(
                    Block(
                        block_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        source_file=result.source_file,
                        block_type=BlockType.TABLE,
                        content=markdown,
                        structured_data=rows,
                        order_index=order_index,
                    )
                )
                order_index += 1

        self._link_captions_to_media(result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_block_items(docx_doc: DocxDocument):
        """Yield Paragraph/Table objects in the order they appear in the
        document body (python-docx does not interleave these by default).
        """
        parent_elm = docx_doc.element.body
        for child in parent_elm.iterchildren():
            if child.tag == qn("w:p"):
                yield Paragraph(child, docx_doc)
            elif child.tag == qn("w:tbl"):
                yield Table(child, docx_doc)

    @staticmethod
    def _image_rids_in_paragraph(paragraph: Paragraph) -> list[str]:
        rids = []
        blips = paragraph._p.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
        )
        for blip in blips:
            rid = blip.get(qn("r:embed"))
            if rid:
                rids.append(rid)
        return rids

    @staticmethod
    def _run_ocr(image_path: str, warnings: list) -> str:
        try:
            with Image.open(image_path) as im:
                return pytesseract.image_to_string(im)
        except Exception as e:
            warnings.append(f"OCR failed for {image_path}: {e}")
            return ""

    @staticmethod
    def _table_to_markdown(table: Table):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if not rows:
            return "", rows
        header, *body = rows
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines), rows

    @staticmethod
    def _link_captions_to_media(result: ExtractionResult):
        captions = [b for b in result.blocks if b.block_type == BlockType.CAPTION]
        media = [b for b in result.blocks if b.block_type in (BlockType.IMAGE, BlockType.TABLE)]

        for caption in captions:
            nearby = [m for m in media if abs(m.order_index - caption.order_index) <= CAPTION_LINK_WINDOW]
            if not nearby:
                continue
            nearest = min(nearby, key=lambda m: abs(m.order_index - caption.order_index))
            result.link(caption, nearest)