"""
app/extraction/csv_extractor.py

CSV extraction: the whole file is treated as one table, split into
row-chunked TABLE blocks (same MAX_ROWS_PER_TABLE_BLOCK convention as
xlsx_extractor.py) so a single block stays embeddable. A HEADING block
carrying the filename anchors the chunks together via `relates_to`.

Does not touch app/ingestion/pipeline.py.
"""

from __future__ import annotations

import csv
import os
import uuid

from app.extraction.base import (
    BaseExtractor,
    Block,
    BlockType,
    ExtractionResult,
)

MAX_ROWS_PER_TABLE_BLOCK = 40


class CsvExtractor(BaseExtractor):
    file_types = ("csv", "tsv")

    def extract(self, file_path: str, doc_id: str) -> ExtractionResult:
        result = ExtractionResult(
            doc_id=doc_id,
            source_file=os.path.basename(file_path),
            file_type="csv",
        )

        rows = self._read_rows(file_path, result.warnings)
        if not rows:
            result.warnings.append("No rows found in CSV file.")
            return result

        order_index = 0

        heading_block = Block(
            block_id=str(uuid.uuid4()),
            doc_id=doc_id,
            source_file=result.source_file,
            block_type=BlockType.HEADING,
            content=f"File: {result.source_file}",
            order_index=order_index,
        )
        result.blocks.append(heading_block)
        order_index += 1

        header, body = rows[0], rows[1:]

        for start in range(0, max(len(body), 1), MAX_ROWS_PER_TABLE_BLOCK):
            chunk = body[start:start + MAX_ROWS_PER_TABLE_BLOCK]
            if not chunk and start > 0:
                break
            markdown = self._rows_to_markdown(header, chunk)

            result.blocks.append(
                Block(
                    block_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    source_file=result.source_file,
                    block_type=BlockType.TABLE,
                    content=markdown,
                    structured_data=[header] + chunk,
                    order_index=order_index,
                    extra_metadata={
                        "row_range": [start + 2, start + 1 + len(chunk)],  # 1-indexed, +header row
                    },
                    relates_to=[heading_block.block_id],
                )
            )
            heading_block.relates_to.append(result.blocks[-1].block_id)
            order_index += 1

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_rows(file_path: str, warnings: list) -> list[list[str]]:
        try:
            with open(file_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except csv.Error:
                    dialect = csv.excel
                reader = csv.reader(f, dialect)
                return [row for row in reader if any(cell.strip() for cell in row)]
        except Exception as e:
            warnings.append(f"Failed to read CSV: {e}")
            return []

    @staticmethod
    def _rows_to_markdown(header: list[str], body: list[list[str]]) -> str:
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for row in body:
            padded = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(padded[:len(header)]) + " |")
        return "\n".join(lines)