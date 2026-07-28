"""
app/chunking/relational_chunker.py

Relationship-aware chunking.

The existing splitter (app/ingestion/pipeline.py::split_chunks) uses a flat
RecursiveCharacterTextSplitter(500/100) over plain text -- it has no concept
of tables, images, or captions, and is untouched by this file.

This chunker instead works over Block objects (see app/extraction/base.py),
which already carry `relates_to` links from the extractors (e.g. a table
linked to its caption, an image linked to its OCR text and caption). It:

  1. Finds connected groups of related blocks (a table+caption+heading is
     one group; a standalone paragraph is a group of one).
  2. For plain running text, chunks sequentially with a size target and
     overlap, same spirit as the existing splitter.
  3. For a relational group (table/image + its caption/heading context),
     emits one chunk per table/image, each PREFIXED with the shared heading/
     caption text -- so a retrieved table chunk is never context-less, and
     a retrieved image's OCR/caption stays attached to the image_path.

Output is a list of langchain_core.documents.Document, so it drops into the
existing FAISS/BM25/HybridRetriever code (app/retrieval/*) with zero changes
there.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.extraction.base import Block, BlockType

MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 100

CONTEXT_TYPES = {BlockType.HEADING, BlockType.TEXT, BlockType.CAPTION}
MEDIA_TYPES = {BlockType.TABLE, BlockType.IMAGE, BlockType.OCR_TEXT}


class RelationalChunker:
    def __init__(self, max_chunk_chars: int = MAX_CHUNK_CHARS, overlap_chars: int = OVERLAP_CHARS):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_chars, chunk_overlap=overlap_chars
        )

    def chunk(self, blocks: list[Block]) -> list[Document]:
        if not blocks:
            return []

        groups = self._connected_groups(blocks)

        linear_blocks: list[Block] = []
        relational_groups: list[list[Block]] = []

        for group in groups:
            if len(group) == 1 and group[0].block_type in CONTEXT_TYPES:
                linear_blocks.append(group[0])
            else:
                relational_groups.append(sorted(group, key=lambda b: b.order_index))

        documents: list[Document] = []
        documents.extend(self._chunk_linear_text(linear_blocks))
        for group in relational_groups:
            documents.extend(self._chunk_relational_group(group))

        return documents

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _connected_groups(blocks: list[Block]) -> list[list[Block]]:
        """Union-find over `relates_to` edges. A block with no relations is
        its own group of size 1.
        """
        by_id = {b.block_id: b for b in blocks}
        parent = {b.block_id: b.block_id for b in blocks}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for b in blocks:
            for related_id in b.relates_to:
                if related_id in by_id:
                    union(b.block_id, related_id)

        groups: dict[str, list[Block]] = {}
        for b in blocks:
            root = find(b.block_id)
            groups.setdefault(root, []).append(b)

        return list(groups.values())

    # ------------------------------------------------------------------
    # Linear text chunking (headings/paragraphs/standalone captions)
    # ------------------------------------------------------------------

    def _chunk_linear_text(self, blocks: list[Block]) -> list[Document]:
        if not blocks:
            return []

        blocks = sorted(blocks, key=lambda b: b.order_index)
        documents: list[Document] = []

        buffer_blocks: list[Block] = []
        buffer_len = 0

        def flush():
            if not buffer_blocks:
                return
            combined_text = "\n\n".join(b.content for b in buffer_blocks)
            for piece in self._text_splitter.split_text(combined_text):
                documents.append(
                    Document(
                        page_content=piece,
                        metadata=self._merged_metadata(buffer_blocks, chunk_type="text"),
                    )
                )

        for block in blocks:
            block_len = len(block.content)
            if buffer_blocks and buffer_len + block_len > self.max_chunk_chars:
                flush()
                buffer_blocks = []
                buffer_len = 0
            buffer_blocks.append(block)
            buffer_len += block_len

        flush()
        return documents

    # ------------------------------------------------------------------
    # Relational group chunking (table/image + surrounding context)
    # ------------------------------------------------------------------

    def _chunk_relational_group(self, group: list[Block]) -> list[Document]:
        context_blocks = [b for b in group if b.block_type in CONTEXT_TYPES]
        media_blocks = [b for b in group if b.block_type in MEDIA_TYPES]

        context_prefix = "\n".join(b.content for b in context_blocks if b.content.strip())

        if not media_blocks:
            # Group of only context blocks that happened to be linked
            # (e.g. a caption linked to a heading with no media found) --
            # treat as plain text.
            return self._chunk_linear_text(context_blocks)

        documents: list[Document] = []
        sibling_ids = [b.block_id for b in media_blocks]

        for media in media_blocks:
            if media.block_type == BlockType.TABLE:
                body = media.content
            elif media.block_type == BlockType.OCR_TEXT:
                body = f"[Image text (OCR)]\n{media.content}"
            else:  # IMAGE with no OCR text of its own -- content lives in linked OCR/caption blocks
                continue  # avoid emitting an empty-content chunk; OCR/caption blocks cover this

            parts = [p for p in (context_prefix, body) if p.strip()]
            page_content = "\n\n".join(parts)

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=self._merged_metadata(
                        context_blocks + [media],
                        chunk_type=media.block_type.value,
                        sibling_block_ids=[i for i in sibling_ids if i != media.block_id],
                        image_path=self._find_image_path(group),
                    ),
                )
            )

        # If the group is an image with only a caption (no OCR text, no
        # table), the caption itself carries the retrievable content --
        # already covered by context_prefix. Emit one chunk for that case.
        if not documents and context_prefix.strip():
            image_block = next((b for b in media_blocks if b.block_type == BlockType.IMAGE), None)
            documents.append(
                Document(
                    page_content=context_prefix,
                    metadata=self._merged_metadata(
                        group, chunk_type="image_caption_only",
                        image_path=image_block.image_path if image_block else None,
                    ),
                )
            )

        return documents

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _find_image_path(group: list[Block]) -> str | None:
        for b in group:
            if b.block_type == BlockType.IMAGE and b.image_path:
                return b.image_path
        return None

    @staticmethod
    def _merged_metadata(
        blocks: list[Block],
        chunk_type: str,
        sibling_block_ids: list[str] | None = None,
        image_path: str | None = None,
    ) -> dict:
        first = blocks[0]
        return {
            "doc_id": first.doc_id,
            "source": first.source_file,
            "chunk_type": chunk_type,
            "block_ids": [b.block_id for b in blocks],
            "block_types": list({b.block_type.value for b in blocks}),
            "page_numbers": sorted({b.page_number for b in blocks if b.page_number is not None}),
            "sibling_block_ids": sibling_block_ids or [],
            "image_path": image_path,
        }