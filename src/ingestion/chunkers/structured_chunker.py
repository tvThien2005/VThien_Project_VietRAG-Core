"""Token-bounded chunking over structured parser blocks."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer

from src.config import get_config
from src.ingestion.chunkers.section_builder import SectionBuilder
from src.logger import get_logger
from src.schemas.blocks import BlockType, ParsedBlock, TableData
from src.schemas.document import Chunk

logger = get_logger("structured_chunker")

_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
_EXPLICIT_CHAPTER = re.compile(r"^(?:chương|chapter)\s+\d+\b", re.IGNORECASE)
_RECURSIVE_SEPARATORS = ("\n\n", "\n", "; ", ", ", " ")


@dataclass
class _Piece:
    text: str
    source_text: str
    block: ParsedBlock
    start: int
    end: int


class StructuredChunker:
    """Build chunks from block hierarchy; tables remain indivisible units."""

    version = "structured-v1"

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
        *,
        tokenizer: Any = None,
        tokenizer_name: Optional[str] = None,
    ):
        cfg = get_config()
        self.max_tokens = (
            max_tokens if max_tokens is not None else cfg.chunking.structured.max_tokens
        )
        self.overlap_tokens = (
            overlap_tokens
            if overlap_tokens is not None
            else cfg.chunking.structured.overlap_tokens
        )
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens must be non-negative and less than max_tokens.")

        self.tokenizer_name = tokenizer_name or cfg.embedding.model_name
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=True,
            )
        )
        self.section_builder = SectionBuilder()
        self._section_chunk_indices: Dict[Optional[str], int] = {}

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _sentence_pieces(text: str) -> List[Tuple[str, int, int]]:
        if not text:
            return []
        pieces: List[Tuple[str, int, int]] = []
        cursor = 0
        for match in _SENTENCE_END.finditer(text):
            end = match.end()
            if end > cursor:
                pieces.append((text[cursor:end], cursor, end))
                cursor = end
        if cursor < len(text):
            pieces.append((text[cursor:], cursor, len(text)))
        return pieces or [(text, 0, len(text))]

    def _token_windows(
        self,
        text: str,
        max_tokens: int,
    ) -> List[Tuple[str, int, int]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        if offsets and isinstance(offsets[0], list):
            offsets = offsets[0]
        offsets = [(int(start), int(end)) for start, end in offsets if end > start]
        if len(offsets) <= max_tokens:
            return [(text, 0, len(text))]

        overlap = min(self.overlap_tokens, max_tokens - 1)
        step = max_tokens - overlap
        windows: List[Tuple[str, int, int]] = []
        token_start = 0
        previous_end = 0
        while token_start < len(offsets):
            token_end = min(token_start + max_tokens, len(offsets))
            char_start = 0 if not windows else offsets[token_start][0]
            if not overlap and windows:
                char_start = previous_end
            char_end = len(text) if token_end == len(offsets) else offsets[token_end - 1][1]
            if char_end <= char_start:
                raise ValueError("Tokenizer returned invalid offsets while splitting prose.")
            windows.append((text[char_start:char_end], char_start, char_end))
            previous_end = char_end
            if token_end == len(offsets):
                break
            token_start += step
        return windows

    def _recursive_windows(
        self,
        text: str,
        max_tokens: int,
    ) -> List[Tuple[str, int, int]]:
        """Recursively split over structural punctuation before token offsets."""

        def split_range(start: int, end: int, separator_index: int) -> List[Tuple[str, int, int]]:
            fragment = text[start:end]
            if self.count_tokens(fragment) <= max_tokens:
                return [(fragment, start, end)]

            for index in range(separator_index, len(_RECURSIVE_SEPARATORS)):
                separator = _RECURSIVE_SEPARATORS[index]
                matches = list(re.finditer(re.escape(separator), fragment))
                if not matches:
                    continue

                ranges: List[Tuple[int, int]] = []
                cursor = start
                for match in matches:
                    boundary = start + match.end()
                    if boundary > cursor:
                        ranges.append((cursor, boundary))
                    cursor = boundary
                if cursor < end:
                    ranges.append((cursor, end))
                if len(ranges) < 2:
                    continue

                atoms: List[Tuple[str, int, int]] = []
                for atom_start, atom_end in ranges:
                    atoms.extend(split_range(atom_start, atom_end, index + 1))

                packed: List[Tuple[str, int, int]] = []
                current_start: Optional[int] = None
                current_end = 0
                current_text = ""
                for atom_text, atom_start, atom_end in atoms:
                    candidate = current_text + atom_text
                    if current_text and self.count_tokens(candidate) > max_tokens:
                        packed.append((current_text, current_start or 0, current_end))
                        current_start = atom_start
                        current_text = atom_text
                    else:
                        if current_start is None:
                            current_start = atom_start
                        current_text = candidate
                    current_end = atom_end
                if current_text:
                    packed.append((current_text, current_start or 0, current_end))
                return packed

            return [
                (window, start + left, start + right)
                for window, left, right in self._token_windows(fragment, max_tokens)
            ]

        return split_range(0, len(text), 0)

    @staticmethod
    def _render_heading(block: ParsedBlock) -> str:
        level = block.heading_level or 1
        return f"{'#' * level} {block.text or ''}"

    @staticmethod
    def _render_table(table: TableData) -> str:
        lines: List[str] = []
        if table.caption:
            lines.append(table.caption)
        rows = [*table.header_rows, *table.rows]
        if table.header_rows:
            for index, row in enumerate(rows):
                padded = row + [""] * (len(table.columns) - len(row))
                lines.append("| " + " | ".join(padded) + " |")
                if index == len(table.header_rows) - 1:
                    lines.append("| " + " | ".join(["---"] * len(table.columns)) + " |")
        else:
            for index, row in enumerate(table.rows, start=1):
                lines.append(f"Row {index}: " + " | ".join(row))
        return "\n".join(lines)

    @staticmethod
    def _source_table_text(table: TableData) -> str:
        return "\n".join(
            cell.text
            for row in table.cells
            for cell in row
            if cell.text
        )

    def _next_section_index(self, section_id: Optional[str]) -> int:
        index = self._section_chunk_indices.get(section_id, 0)
        self._section_chunk_indices[section_id] = index + 1
        return index

    def _create_chunk(
        self,
        pieces: Sequence[_Piece],
        *,
        doc_id: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        table: Optional[ParsedBlock] = None,
        oversize_reason: Optional[str] = None,
    ) -> Chunk:
        if not pieces:
            raise ValueError("Cannot create a structured chunk without source pieces.")
        text = "".join(piece.text for piece in pieces).strip()
        if not text:
            raise ValueError("Cannot create an empty structured chunk.")

        first_block = pieces[0].block
        section_path = list(first_block.section_path)
        section_id = first_block.section_id
        block_ids = list(dict.fromkeys(piece.block.block_id for piece in pieces))
        for piece in pieces:
            caption_block_id = piece.block.metadata.get("caption_block_id")
            if caption_block_id and caption_block_id not in block_ids:
                block_ids.append(caption_block_id)
        block_types = list(dict.fromkeys(piece.block.block_type.value for piece in pieces))
        pages = [
            piece.block.page_number
            for piece in pieces
            if piece.block.page_number is not None
        ]
        source_fragments = [
            {
                "block_id": piece.block.block_id,
                "start_char": piece.start,
                "end_char": piece.end,
                "content": piece.source_text,
            }
            for piece in pieces
            if piece.source_text
        ]
        chunk_id = str(uuid.uuid4())
        source = first_block.metadata.get("source", "")
        metadata: Dict[str, Any] = {
            **first_block.metadata,
            "source": source,
            "file_name": first_block.metadata.get("file_name", ""),
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "section_path": section_path,
            "section_id": section_id,
            "parent_section_id": first_block.metadata.get("parent_section_id"),
            "chapter": (
                section_path[0]
                if section_path and _EXPLICIT_CHAPTER.match(section_path[0])
                else ""
            ),
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "block_ids": block_ids,
            "source_block_ids": block_ids,
            "block_types": block_types,
            "source_fragments": source_fragments,
            "source_content": "\n\n".join(fragment["content"] for fragment in source_fragments),
            "chunking_version": self.version,
            "tokenizer": self.tokenizer_name,
            "token_count": self.count_tokens(text),
            "token_limit": self.max_tokens,
            "token_limit_exceeded": self.count_tokens(text) > self.max_tokens,
            "chunk_has_table": table is not None,
            "chunk_index_in_section": self._next_section_index(section_id),
        }
        if pages and min(pages) == max(pages):
            metadata["page_number"] = min(pages)
        if table is not None and table.table is not None:
            metadata["source_table"] = table.table.model_dump(mode="json")
            metadata["table_id"] = table.table.table_id
            metadata["table_caption"] = table.table.caption
            metadata["table_caption_status"] = (
                "present"
                if table.table.caption and table.table.caption.strip()
                else "missing"
            )
        if oversize_reason:
            metadata["oversize_reason"] = oversize_reason
        if extra_metadata:
            metadata.update(extra_metadata)

        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            chunk_index=0,
            metadata=metadata,
        )

    def _split_oversized_piece(
        self,
        piece: _Piece,
        doc_id: str,
    ) -> List[Chunk]:
        windows = self._recursive_windows(piece.text, self.max_tokens)
        chunks: List[Chunk] = []
        for window_text, start, end in windows:
            source_start = piece.start + start
            source_end = piece.start + end
            window = _Piece(
                text=window_text,
                source_text=piece.source_text[start:end],
                block=piece.block,
                start=source_start,
                end=source_end,
            )
            chunks.append(
                self._create_chunk(
                    [window],
                    doc_id=doc_id,
                    extra_metadata={"overlap_applied": bool(chunks and self.overlap_tokens)},
                )
            )
        return chunks

    def split_blocks(
        self,
        blocks: Sequence[ParsedBlock],
        *,
        doc_id: Optional[str] = None,
    ) -> List[Chunk]:
        if not blocks:
            return []
        built = self.section_builder.build(blocks)
        document_id = doc_id or str(uuid.uuid4())
        self._section_chunk_indices = {}
        chunks: List[Chunk] = []
        prose: List[_Piece] = []
        pending_headings: List[ParsedBlock] = []
        active_section_id: Optional[str] = None
        has_active_section = False
        last_prose_block_id: Optional[str] = None

        def flush_prose() -> None:
            nonlocal prose, last_prose_block_id
            while prose:
                current: List[_Piece] = []
                cursor = 0
                while cursor < len(prose):
                    piece = prose[cursor]
                    candidate = [*current, piece]
                    if current and self.count_tokens("".join(part.text for part in candidate)) > self.max_tokens:
                        chunks.append(
                            self._create_chunk(current, doc_id=document_id)
                        )
                        current = []
                        continue

                    if not current and self.count_tokens(piece.text) > self.max_tokens:
                        chunks.extend(self._split_oversized_piece(piece, document_id))
                        cursor += 1
                        continue

                    current.append(piece)
                    cursor += 1
                if current:
                    chunks.append(self._create_chunk(current, doc_id=document_id))
                prose = []
            last_prose_block_id = None

        def pending_heading_pieces() -> Iterator[_Piece]:
            for heading in pending_headings:
                yield_piece = _Piece(
                    text=self._render_heading(heading) + "\n\n",
                    source_text=heading.text or "",
                    block=heading,
                    start=0,
                    end=len(heading.text or ""),
                )
                yield yield_piece

        def add_sentence_piece(piece: _Piece) -> None:
            nonlocal pending_headings, last_prose_block_id
            headings = list(pending_heading_pieces())
            separator = (
                [
                    _Piece(
                        text="\n\n",
                        source_text="",
                        block=piece.block,
                        start=piece.start,
                        end=piece.start,
                    )
                ]
                if prose and last_prose_block_id != piece.block.block_id
                else []
            )
            candidate = [*prose, *separator, *headings, piece]
            if prose and self.count_tokens("".join(item.text for item in candidate)) > self.max_tokens:
                flush_prose()
                separator = []
                candidate = [*headings, piece]

            if self.count_tokens("".join(item.text for item in candidate)) <= self.max_tokens:
                prose.extend(separator)
                prose.extend(headings)
                prose.append(piece)
                pending_headings = []
                last_prose_block_id = piece.block.block_id
                return

            if prose:
                flush_prose()
            prefix = "".join(item.text for item in headings)
            available = self.max_tokens - self.count_tokens(prefix)
            if headings and available < 1:
                heading_chunks = [
                    self._create_chunk(
                        [heading_piece],
                        doc_id=document_id,
                        extra_metadata={
                            "heading_only": True,
                            "oversize_reason": "heading_exceeds_token_budget",
                        },
                    )
                    for heading_piece in headings
                ]
                chunks.extend(heading_chunks)
                headings = []
                prefix = ""
                available = self.max_tokens

            windows = self._recursive_windows(piece.text, available)
            for window_index, (window_text, start_offset, end_offset) in enumerate(windows):
                window_piece = _Piece(
                    text=window_text,
                    source_text=piece.source_text[start_offset:end_offset],
                    block=piece.block,
                    start=piece.start + start_offset,
                    end=piece.start + end_offset,
                )
                current = [*headings, window_piece] if window_index == 0 else [window_piece]
                if self.count_tokens("".join(item.text for item in current)) > self.max_tokens:
                    chunks.extend(
                        self._split_oversized_piece(window_piece, document_id)
                    )
                else:
                    chunks.append(
                        self._create_chunk(
                            current,
                            doc_id=document_id,
                            extra_metadata={
                                "overlap_applied": bool(
                                    window_index > 0 and self.overlap_tokens
                                )
                            },
                        )
                    )
            pending_headings = []
            last_prose_block_id = None

        caption_to_table: Dict[str, str] = {}
        for block in built.blocks:
            if block.block_type == BlockType.CAPTION and block.metadata.get("table_block_id"):
                caption_to_table[block.block_id] = block.metadata["table_block_id"]

        for index, block in enumerate(built.blocks):
            if has_active_section and block.section_id != active_section_id:
                flush_prose()
                if pending_headings:
                    heading_only_pieces = [
                        _Piece(
                            text=self._render_heading(heading),
                            source_text=heading.text or "",
                            block=heading,
                            start=0,
                            end=len(heading.text or ""),
                        )
                        for heading in pending_headings
                    ]
                    chunks.append(
                        self._create_chunk(
                            heading_only_pieces,
                            doc_id=document_id,
                            extra_metadata={"heading_only": True},
                        )
                    )
                    pending_headings = []
                last_prose_block_id = None
            active_section_id = block.section_id
            has_active_section = True

            if block.block_type == BlockType.HEADING:
                flush_prose()
                pending_headings.append(block)
                continue
            if block.block_type == BlockType.CAPTION:
                linked_table_id = caption_to_table.get(block.block_id)
                if linked_table_id:
                    continue
                next_block = built.blocks[index + 1] if index + 1 < len(built.blocks) else None
                if (
                    next_block is not None
                    and next_block.block_type == BlockType.TABLE
                    and next_block.metadata.get("caption_block_id") == block.block_id
                ):
                    continue

            if block.block_type == BlockType.TABLE:
                flush_prose()
                table_text = self._render_table(block.table) if block.table else ""
                table_source = self._source_table_text(block.table) if block.table else ""
                if block.table and block.table.caption:
                    table_source = f"{block.table.caption}\n{table_source}".strip()
                table_piece = _Piece(
                    text=table_text,
                    source_text=table_source,
                    block=block,
                    start=0,
                    end=len(table_source),
                )
                table_pieces: List[_Piece] = []
                for heading in pending_headings:
                    table_pieces.append(
                        _Piece(
                            text=self._render_heading(heading) + "\n\n",
                            source_text=heading.text or "",
                            block=heading,
                            start=0,
                            end=len(heading.text or ""),
                        )
                    )
                pending_headings = []
                table_pieces.append(table_piece)
                token_count = self.count_tokens("".join(piece.text for piece in table_pieces))
                chunks.append(
                    self._create_chunk(
                        table_pieces,
                        doc_id=document_id,
                        table=block,
                        oversize_reason=(
                            "table_is_an_indivisible_structural_unit"
                            if token_count > self.max_tokens
                            else None
                        ),
                    )
                )
                continue

            text = block.text or ""
            if not text:
                continue
            is_textual = block.block_type in {
                BlockType.PARAGRAPH,
                BlockType.LIST,
                BlockType.EQUATION,
                BlockType.CODE,
                BlockType.CAPTION,
                BlockType.FIGURE,
            }
            if not is_textual:
                continue

            pieces = self._sentence_pieces(text) if block.block_type == BlockType.PARAGRAPH else [(text, 0, len(text))]
            for sentence, start, end in pieces:
                piece = _Piece(
                    text=sentence,
                    source_text=sentence,
                    block=block,
                    start=start,
                    end=end,
                )
                add_sentence_piece(piece)

        flush_prose()
        if pending_headings:
            heading_pieces = [
                _Piece(
                    text=self._render_heading(heading),
                    source_text=heading.text or "",
                    block=heading,
                    start=0,
                    end=len(heading.text or ""),
                )
                for heading in pending_headings
            ]
            chunks.append(
                self._create_chunk(
                    heading_pieces,
                    doc_id=document_id,
                    extra_metadata={"heading_only": True},
                )
            )

        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
            chunk.metadata["chunk_index"] = index
        logger.info(
            "Structured chunking produced %s chunks from %s blocks (max_tokens=%s, tokenizer=%s)",
            len(chunks),
            len(built.blocks),
            self.max_tokens,
            self.tokenizer_name,
        )
        return chunks


__all__ = ["StructuredChunker"]
