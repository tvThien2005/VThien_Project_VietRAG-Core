"""Adapters between normalized parsed blocks and the legacy Document API."""

from __future__ import annotations

from typing import Dict, Iterable, List

from src.schemas.blocks import BlockType, ParsedBlock
from src.schemas.document import Document
from src.utils.text import clean_text


def render_block_for_legacy_text(
    block: ParsedBlock,
    *,
    include_table_caption: bool = True,
) -> str:
    """Render a block only at the legacy boundary, never as parser storage."""
    if block.block_type == BlockType.TABLE:
        if block.table is None:
            raise ValueError(f"TABLE block {block.block_id} thiếu TableData")
        
        prefix = f"*{block.table.caption}*\n" if include_table_caption and block.table.caption else ""
        # Legacy text consumers expect a Markdown header.  The normalized
        # representation itself does not assume the first data row is a header.
        if not block.table.header and block.table.rows:
            rows = block.table.rows
            header = rows[0]
            lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * len(header)) + " |",
                *["| " + " | ".join(row) + " |" for row in rows[1:]],
            ]
            return prefix + "\n".join(lines)
        return prefix + block.table.to_markdown()
    if block.block_type == BlockType.HEADING:
        return f"{'#' * (block.heading_level or 1)} {block.text or ''}".strip()
    return block.text or ""


def documents_from_blocks(blocks: Iterable[ParsedBlock], *, clean_content: bool) -> List[Document]:
    """Return the existing ``List[Document]`` contract from parsed blocks.

    A legacy document remains bounded by page and heading, matching the prior
    loaders' observable behavior.  Block IDs are retained as a lightweight
    link; full structured blocks are obtained through ``load_blocks`` and are
    intentionally not copied into every legacy chunk/Qdrant payload.
    """
    documents: List[Document] = []
    pending: List[ParsedBlock] = []
    pending_page = None

    def flush() -> None:
        nonlocal pending, pending_page
        if not pending:
            return
        rendered = []
        for index, block in enumerate(pending):
            preceding_caption = (
                index > 0
                and pending[index - 1].block_type == BlockType.CAPTION
                and block.block_type == BlockType.TABLE
                and block.table is not None
                and block.table.caption == pending[index - 1].text
            )
            following_caption = (
                index + 1 < len(pending)
                and block.block_type == BlockType.TABLE
                and block.table is not None
                and pending[index + 1].block_type == BlockType.CAPTION
                and block.table.caption == pending[index + 1].text
            )
            rendered.append(
                render_block_for_legacy_text(
                    block,
                    include_table_caption=not (preceding_caption or following_caption),
                )
            )
        raw = "\n".join(rendered)
        text = clean_text(raw) if clean_content else raw.strip()
        if text:
            last = pending[-1]
            metadata: Dict = dict(last.metadata)
            metadata.update(
                {
                    "parsed_block_ids": [block.block_id for block in pending],
                    "section_id": last.section_id,
                    "section_path": list(last.section_path),
                }
            )
            if last.page_number is not None:
                metadata["page_number"] = last.page_number
            documents.append(Document(text=text, metadata=metadata))
        pending = []
        pending_page = None

    for block in blocks:
        page_changed = (
            pending
            and block.page_number is not None
            and pending_page is not None
            and block.page_number != pending_page
        )
        if page_changed or (pending and block.block_type == BlockType.HEADING):
            flush()
        pending.append(block)
        if block.page_number is not None:
            pending_page = block.page_number

    flush()
    return documents
