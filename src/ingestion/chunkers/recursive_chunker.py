import re
from typing import Any, List, Optional, Tuple
import uuid

from src.config import get_config
from src.constants import VIETNAMESE_SEPARATORS
from src.logger import get_logger
from src.schemas.document import Chunk, Document

logger = get_logger("recursive_chunker")

_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
_TABLE_CAPTION = re.compile(r"^(?:bảng|table)(?:\s|\d|[:.)-]|$)", re.IGNORECASE)


class RecursiveCharacterChunker:
    """Bộ chia nhỏ văn bản đệ quy (Recursive Character Chunker).
    
    Thuật toán ưu tiên chia theo các cấu trúc ngữ nghĩa lớn (đoạn văn, dòng mới)
    trước khi chia theo dấu chấm câu, dấu phẩy và khoảng trắng từ trong Tiếng Việt.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ):
        cfg = get_config()
        self.chunk_size = chunk_size or cfg.chunking.recursive.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunking.recursive.chunk_overlap
        self.separators = separators or cfg.chunking.recursive.separators or VIETNAMESE_SEPARATORS

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) phải nhỏ hơn chunk_size ({self.chunk_size})"
            )

    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        """Chia văn bản theo 1 separator cụ thể và giữ lại separator đó."""
        if not separator:
            return list(text)
        splits = text.split(separator)
        return [s for s in splits if s]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Hàm đệ quy chia văn bản theo danh sách separators."""
        final_chunks: List[str] = []

        # 1. Tìm separator đầu tiên xuất hiện trong văn bản
        chosen_sep = ""
        new_separators: List[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                chosen_sep = ""
                break
            if sep in text:
                chosen_sep = sep
                new_separators = separators[i + 1 :]
                break

        # 2. Tách text theo chosen_sep
        if chosen_sep != "":
            splits = self._split_text_with_separator(text, chosen_sep)
        else:
            splits = list(text)

        # 3. Gom các đoạn con lại thành chunk có kích thước <= chunk_size
        good_splits: List[str] = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                # Nếu đoạn con vẫn lớn hơn chunk_size và còn separator con, gọi đệ quy tiếp
                if good_splits:
                    merged = self._merge_splits(good_splits, chosen_sep)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    sub_chunks = self._recursive_split(s, new_separators)
                    final_chunks.extend(sub_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, chosen_sep)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Gom các đoạn nhỏ lại kết hợp xử lý overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        for s in splits:
            s_len = len(s)
            sep_len = len(separator) if current_doc else 0

            if total_len + s_len + sep_len > self.chunk_size:
                if total_len > 0:
                    joined_text = separator.join(current_doc).strip()
                    if joined_text:
                        docs.append(joined_text)

                    # Tạo overlap: lùi lại các phần tử cho đến khi độ dài <= chunk_overlap
                    while current_doc and (
                        total_len > self.chunk_overlap
                        or total_len + s_len + (len(separator) if current_doc else 0) > self.chunk_size
                    ):
                        removed = current_doc.pop(0)
                        total_len -= len(removed) + (len(separator) if current_doc else 0)

            current_doc.append(s)
            total_len += s_len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            joined_text = separator.join(current_doc).strip()
            if joined_text:
                docs.append(joined_text)

        return docs

    def split_text(self, text: str) -> List[str]:
        """Cắt một chuỗi văn bản thành danh sách các chuỗi chunk nhỏ."""
        if not text or not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]

        lines = text.splitlines()
        table_ranges = self._find_table_ranges(lines)
        if table_ranges:
            chunks: List[str] = []
            cursor = 0
            for (
                start,
                end,
                header_index,
                separator_index,
                caption_after_index,
            ) in table_ranges:
                prose = "\n".join(lines[cursor:start]).strip()
                if prose:
                    chunks.extend(self._recursive_split(prose, self.separators))
                chunks.extend(
                    self._split_table(
                        lines[start:end],
                        header_index - start,
                        separator_index - start,
                        (
                            caption_after_index - start
                            if caption_after_index is not None
                            else None
                        ),
                    )
                )
                cursor = end

            trailing_prose = "\n".join(lines[cursor:]).strip()
            if trailing_prose:
                chunks.extend(self._recursive_split(trailing_prose, self.separators))
            return chunks

        return self._recursive_split(text, self.separators)

    @staticmethod
    def _is_table_caption(line: str) -> bool:
        caption = line.strip().strip("*_ ").strip()
        return bool(_TABLE_CAPTION.match(caption))

    def _find_table_ranges(
        self,
        lines: List[str],
    ) -> List[Tuple[int, int, int, int, Optional[int]]]:
        """Find Markdown tables and include adjacent captions before or after them."""
        ranges: List[Tuple[int, int, int, int, Optional[int]]] = []
        line_index = 1
        while line_index < len(lines) - 1:
            if (
                not _TABLE_SEPARATOR.match(lines[line_index])
                or "|" not in lines[line_index]
                or "|" not in lines[line_index - 1]
            ):
                line_index += 1
                continue

            header_index = line_index - 1
            end = line_index + 1
            while end < len(lines) and lines[end].strip() and "|" in lines[end]:
                end += 1

            start = header_index
            caption_index = header_index - 1
            while caption_index >= 0:
                while caption_index >= 0 and not lines[caption_index].strip():
                    caption_index -= 1
                if caption_index >= 0 and self._is_table_caption(lines[caption_index]):
                    start = caption_index
                    caption_index -= 1
                else:
                    break

            caption_after_index: Optional[int] = None
            caption_candidate = end
            while caption_candidate < len(lines) and not lines[caption_candidate].strip():
                caption_candidate += 1
            if (
                caption_candidate < len(lines)
                and self._is_table_caption(lines[caption_candidate])
            ):
                caption_after_index = caption_candidate
                end = caption_candidate + 1

            ranges.append((start, end, header_index, line_index, caption_after_index))
            line_index = end

        return ranges

    def _split_table(
        self,
        lines: List[str],
        header_index: int,
        separator_index: int,
        caption_after_index: Optional[int] = None,
    ) -> List[str]:
        """Split between rows while repeating captions and the header."""
        caption_before = [line for line in lines[:header_index] if line.strip()]
        caption_after = (
            [lines[caption_after_index]]
            if caption_after_index is not None
            else []
        )
        header = lines[header_index : separator_index + 1]
        rows_end = caption_after_index if caption_after_index is not None else len(lines)
        data_rows = [
            line for line in lines[separator_index + 1 : rows_end] if line.strip()
        ]
        prefix = caption_before + caption_after + header
        chunks: List[str] = []
        current_rows: List[str] = []

        for row in data_rows:
            candidate = "\n".join(prefix + current_rows + [row]).strip()
            if current_rows and len(candidate) > self.chunk_size:
                chunks.append("\n".join(prefix + current_rows).strip())
                current_rows = [row]
            else:
                current_rows.append(row)

        if current_rows or not chunks:
            chunks.append("\n".join(prefix + current_rows).strip())
        return chunks

    def split_document(self, document: Document) -> List[Chunk]:
        """Cắt một đối tượng Document thành danh sách các Chunk.
        
        Mỗi chunk được gắn thêm metadata ngữ cảnh:
        - chunk_section_title: tiêu đề mục gần nhất phía trên chunk đó.
        - chunk_breadcrumb: đường dẫn đầy đủ kế thừa từ Document cha (nếu có).
        - chunk_has_table: True nếu chunk chứa dữ liệu bảng Markdown.
        - chunk_index_in_section: vị trí của chunk trong section (0-indexed).
        """
        raw_chunks = self.split_text(document.text)
        chunks: List[Chunk] = []

        # Lấy breadcrumb từ Document cha (nếu loader đã gắn)
        doc_breadcrumb = document.metadata.get("breadcrumb", "")
        doc_section_title = document.metadata.get("section_title", "")
        doc_chapter = document.metadata.get("chapter", "")

        # Theo dõi heading "đang active" khi lướt qua từng chunk
        # Tìm heading (dòng bắt đầu bằng # hoặc ##) trong text của chunk
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        current_section_title = doc_section_title
        chunk_idx_in_section = 0

        for idx, chunk_text in enumerate(raw_chunks):
            # Tìm heading trong chunk này (lấy heading cuối cùng trong chunk làm "active")
            heading_matches = heading_pattern.findall(chunk_text)
            if heading_matches:
                # heading_matches = [(level_hashes, title_text), ...]
                last_heading = heading_matches[-1][1].strip()
                if last_heading:
                    current_section_title = last_heading
                    chunk_idx_in_section = 0  # Reset đếm chunk khi vào section mới
            
            # Kiểm tra chunk có chứa bảng Markdown không
            has_table = any(
                _TABLE_SEPARATOR.match(line) and "|" in chunk_text
                for line in chunk_text.splitlines()
            )

            # Kế thừa toàn bộ metadata từ Document cha
            meta = dict(document.metadata)

            # Bổ sung metadata cụ thể cho chunk
            meta["chunk_section_title"] = current_section_title
            meta["chunk_breadcrumb"] = doc_breadcrumb
            meta["chunk_chapter"] = doc_chapter
            meta["chunk_has_table"] = has_table
            meta["chunk_index_in_section"] = chunk_idx_in_section
            meta["char_count"] = len(chunk_text)

            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=document.doc_id,
                text=chunk_text,
                chunk_index=idx,
                metadata=meta,
            )
            chunks.append(chunk)
            chunk_idx_in_section += 1

        return chunks


    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """Cắt danh sách các Document thành danh sách các Chunk."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Đã cắt {len(documents)} Document thành {len(all_chunks)} Chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks
