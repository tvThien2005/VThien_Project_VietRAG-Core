
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from src.config import get_config
from src.generation.base import BaseLLM
from src.generation.groq_llm import GroqLLM
from src.generation.ollama_llm import OllamaLLM
from src.generation.prompt_builder import PromptBuilder
from src.logger import get_logger
from src.schemas.response import RetrievedChunk

logger = get_logger("generator")


# Các field thường dùng để lưu nội dung chunk.
CONTENT_FIELDS = (
    "text",
    "content",
    "page_content",
    "chunk_text",
)


@dataclass
class FormattedChunk:
    """
    Chunk tối giản được truyền cho PromptBuilder.

    Chỉ giữ:
    - text: nội dung chunk
    - metadata.breadcrumb
    - metadata.file_name
    - metadata.page

    Hỗ trợ cả truy cập bằng thuộc tính và dictionary-style.
    """

    text: str
    metadata: Dict[str, Any]

    @property
    def content(self) -> str:
        """Tương thích với PromptBuilder sử dụng chunk.content."""
        return self.text

    @property
    def page_content(self) -> str:
        """Tương thích với các thành phần sử dụng page_content."""
        return self.text

    def __getitem__(self, key: str) -> Any:
        """Cho phép sử dụng chunk['text'] hoặc chunk['metadata']."""
        if key == "text":
            return self.text

        if key == "content":
            return self.content

        if key == "page_content":
            return self.page_content

        if key == "metadata":
            return self.metadata

        if key in self.metadata:
            return self.metadata[key]

        raise KeyError(key)

    def get(self, key: str, default=None) -> Any:
        """Hỗ trợ cách truy cập tương tự dict.get()."""
        try:
            return self[key]
        except KeyError:
            return default


def get_llm(provider: Optional[str] = None) -> BaseLLM:
    """Khởi tạo LLM provider theo cấu hình config.yaml."""
    cfg = get_config()
    target_provider = (provider or cfg.llm.default_provider).lower()

    if target_provider == "groq":
        return GroqLLM()

    if target_provider == "ollama":
        return OllamaLLM()

    raise ValueError(
        f"LLM Provider '{target_provider}' chưa được hỗ trợ."
    )


class Generator:
    """Điều phối chuẩn hóa chunks, xây dựng prompt và gọi LLM."""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        self.llm = llm or get_llm()
        self.prompt_builder = prompt_builder or PromptBuilder()

    @staticmethod
    def _get_value(item: Any, field: str, default=None) -> Any:
        """Đọc dữ liệu từ dictionary hoặc object."""
        if isinstance(item, dict):
            return item.get(field, default)

        return getattr(item, field, default)

    @classmethod
    def _get_metadata_value(
        cls,
        chunk: Any,
        field: str,
        default=None,
    ) -> Any:
        """
        Đọc metadata từ hai cấu trúc:
        1. chunk.metadata[field]
        2. chunk[field] hoặc thuộc tính chunk.field
        """
        metadata = cls._get_value(chunk, "metadata", {}) or {}

        value = cls._get_value(metadata, field)

        if value is not None:
            return value

        value = cls._get_value(chunk, field)

        return value if value is not None else default

    @classmethod
    def _get_content(cls, chunk: Any) -> str:
        """Lấy nội dung chunk từ field phù hợp."""
        for field in CONTENT_FIELDS:
            content = cls._get_value(chunk, field)

            if isinstance(content, str):
                return content

        return ""

    @staticmethod
    def _normalize_content(content: str) -> str:
        """
        Chuẩn hóa nội dung nhưng giữ cấu trúc đoạn văn,
        tiêu đề Markdown và các dòng trong bảng.
        """
        content = unicodedata.normalize("NFC", content)
        content = content.replace("\u00a0", " ")
        content = content.replace("\r\n", "\n")
        content = content.replace("\r", "\n")

        # Loại bỏ khoảng trắng cuối dòng.
        lines = [line.rstrip() for line in content.split("\n")]
        content = "\n".join(lines)

        # Gộp nhiều dòng trống liên tiếp thành tối đa một dòng trống.
        content = re.sub(
            r"\n[ \t]*\n(?:[ \t]*\n)+",
            "\n\n",
            content,
        )

        return content.strip()

    @staticmethod
    def _make_duplicate_key(content: str) -> str:
        """
        Tạo khóa so sánh nội dung:
        - Không phân biệt chữ hoa và chữ thường.
        - Bỏ khác biệt do khoảng trắng.
        - Chuẩn hóa Unicode.

        Không loại bỏ dấu câu hoặc số liệu để tránh làm mất
        khác biệt quan trọng trong tài liệu kỹ thuật.
        """
        content = unicodedata.normalize("NFC", content)
        content = content.casefold()
        content = re.sub(r"\s+", " ", content)

        return content.strip()

    @classmethod
    def _get_page(cls, chunk: Any) -> Optional[Any]:
        """
        Lấy thông tin trang từ page_start/page_end,
        hoặc từ page/page_number nếu có.

        Không tự suy đoán số trang khi nguồn không cung cấp.
        """
        page_start = cls._get_metadata_value(chunk, "page_start")
        page_end = cls._get_metadata_value(chunk, "page_end")

        if page_start is not None and page_end is not None:
            if page_start == page_end:
                return page_start

            return f"{page_start}-{page_end}"

        if page_start is not None:
            return page_start

        if page_end is not None:
            return page_end

        page = cls._get_metadata_value(chunk, "page")

        if page is not None:
            return page

        return cls._get_metadata_value(chunk, "page_number")

    @classmethod
    def _get_section_path(cls, chunk: Any) -> List[str]:
        """Keep the hierarchy available to the prompt formatter.

        Structured chunks store their hierarchy as a list.  Older payloads can
        contain a single string, so normalize that shape without inventing a
        hierarchy when neither form is present.
        """
        section_path = cls._get_metadata_value(chunk, "section_path", [])
        if isinstance(section_path, str):
            return [section_path.strip()] if section_path.strip() else []
        if isinstance(section_path, (list, tuple)):
            return [str(part).strip() for part in section_path if str(part).strip()]
        return []

    @classmethod
    def _format_chunk(cls, chunk: Any) -> Optional[FormattedChunk]:
        """
        Chuyển chunk ban đầu thành cấu trúc tối giản
        trước khi gửi tới PromptBuilder.
        """
        raw_content = cls._get_content(chunk)
        content = cls._normalize_content(raw_content)

        # Bỏ chunk không có nội dung.
        if not content:
            return None

        metadata = {
            "breadcrumb": (
                cls._get_metadata_value(chunk, "breadcrumb", "") or ""
            ),
            "section_path": cls._get_section_path(chunk),
            "file_name": (
                cls._get_metadata_value(chunk, "file_name", "") or ""
            ),
            "page": cls._get_page(chunk),
        }

        return FormattedChunk(
            text=content,
            metadata=metadata,
        )

    @classmethod
    def _prepare_chunks(
        cls,
        chunks: List[RetrievedChunk],
    ) -> List[FormattedChunk]:
        """
        Chuẩn hóa và loại bỏ chunk trùng lặp.

        Quy tắc:
        1. Giữ nguyên thứ tự đầu vào.
        2. Loại chunk rỗng.
        3. Loại chunk có nội dung trùng sau chuẩn hóa.
        4. Giữ lại metadata breadcrumb, file_name và page.
        5. Khi trùng nội dung, giữ chunk xuất hiện đầu tiên.
        """
        prepared_chunks: List[FormattedChunk] = []
        seen_content = set()

        duplicate_count = 0
        empty_count = 0

        for chunk in chunks:
            formatted_chunk = cls._format_chunk(chunk)

            if formatted_chunk is None:
                empty_count += 1
                continue

            content_key = cls._make_duplicate_key(
                formatted_chunk.text
            )

            if content_key in seen_content:
                duplicate_count += 1
                continue

            seen_content.add(content_key)
            prepared_chunks.append(formatted_chunk)

        logger.info(
            "Chunk preparation: input=%d, output=%d, "
            "duplicates_removed=%d, empty_removed=%d",
            len(chunks),
            len(prepared_chunks),
            duplicate_count,
            empty_count,
        )

        return prepared_chunks

    def generate_answer(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Chuẩn hóa chunks, xây dựng prompt và sinh câu trả lời."""
        prepared_chunks = self._prepare_chunks(chunks)

        sys_prompt, user_prompt = self.prompt_builder.build_prompt(
            query,
            prepared_chunks,
        )

        logger.info(
            "Đang gửi prompt đến LLM để trả lời query: '%s'",
            query,
        )

        return self.llm.generate(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def stream_answer(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Chuẩn hóa chunks và stream câu trả lời từ LLM."""
        prepared_chunks = self._prepare_chunks(chunks)

        sys_prompt, user_prompt = self.prompt_builder.build_prompt(
            query,
            prepared_chunks,
        )

        logger.info(
            "Đang bắt đầu stream câu trả lời cho query: '%s'",
            query,
        )

        return self.llm.stream_generate(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
