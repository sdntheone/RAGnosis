"""
app/extraction/image_extractor.py

Standalone image file extraction (jpg/png/etc uploaded directly, not
embedded in a PDF/DOCX/PPTX/XLSX). Produces:
  - one IMAGE block (points at the saved/copied file)
  - one OCR_TEXT block if pytesseract finds text in the image
  - one CAPTION block if a caption generator is available (see
    app/extraction/caption_generator.py, added separately) -- this uses a
    vision-capable LLM to describe the image content, which matters for
    images that have no embedded text at all (a photo, a chart with no OCR-
    readable labels, etc).

Import of caption_generator is optional/soft: if that module isn't present
yet (e.g. this file was committed first), image extraction still works,
just without generated captions.
"""

from __future__ import annotations

import os
import shutil
import uuid

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
    generate_caption = None  # caption_generator.py not present yet -- degrade gracefully


class ImageExtractor(BaseExtractor):
    file_types = ("png", "jpg", "jpeg", "webp", "bmp", "tiff")

    def __init__(self, image_output_dir: str = "extracted_images"):
        self.image_output_dir = image_output_dir

    def extract(self, file_path: str, doc_id: str) -> ExtractionResult:
        source_file = os.path.basename(file_path)
        result = ExtractionResult(
            doc_id=doc_id,
            source_file=source_file,
            file_type="image",
        )

        doc_image_dir = os.path.join(self.image_output_dir, doc_id)
        os.makedirs(doc_image_dir, exist_ok=True)

        stored_path = os.path.join(doc_image_dir, source_file)
        try:
            if os.path.abspath(stored_path) != os.path.abspath(file_path):
                shutil.copyfile(file_path, stored_path)
        except Exception as e:
            result.warnings.append(f"Failed to store image copy: {e}")
            stored_path = file_path  # fall back to original location

        order_index = 0

        image_block = Block(
            block_id=str(uuid.uuid4()),
            doc_id=doc_id,
            source_file=source_file,
            block_type=BlockType.IMAGE,
            content="",
            image_path=stored_path,
            order_index=order_index,
        )
        result.blocks.append(image_block)
        order_index += 1

        ocr_text = self._run_ocr(stored_path, result.warnings)
        if ocr_text.strip():
            ocr_block = Block(
                block_id=str(uuid.uuid4()),
                doc_id=doc_id,
                source_file=source_file,
                block_type=BlockType.OCR_TEXT,
                content=ocr_text,
                order_index=order_index,
                relates_to=[image_block.block_id],
            )
            image_block.relates_to.append(ocr_block.block_id)
            result.blocks.append(ocr_block)
            order_index += 1

        if generate_caption is not None:
            try:
                caption_text = generate_caption(stored_path)
            except Exception as e:
                caption_text = ""
                result.warnings.append(f"Caption generation failed: {e}")

            if caption_text.strip():
                caption_block = Block(
                    block_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    source_file=source_file,
                    block_type=BlockType.CAPTION,
                    content=caption_text,
                    order_index=order_index,
                    relates_to=[image_block.block_id],
                )
                image_block.relates_to.append(caption_block.block_id)
                result.blocks.append(caption_block)
                order_index += 1
        else:
            result.warnings.append(
                "caption_generator not available -- image captioned only via OCR (if any)."
            )

        return result

    @staticmethod
    def _run_ocr(image_path: str, warnings: list) -> str:
        try:
            with Image.open(image_path) as im:
                return pytesseract.image_to_string(im)
        except Exception as e:
            warnings.append(f"OCR failed for {image_path}: {e}")
            return ""