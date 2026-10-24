
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from src.config import get_config
from src.logger import get_logger
from src.schemas.response import RetrievedChunk

logger = get_logger("prompt_builder")


DEFAULT_SYSTEM_PROMPT = """Bạn là một trợ lý AI thông minh chuyên gia về hỏi đáp tài liệu tiếng Việt.

Nhiệm vụ của bạn là trả lời câu hỏi một cách chính xác, trung thực, rõ ràng và mạch lạc, chỉ dựa trên nội dung của các đoạn ngữ cảnh được cung cấp.

Các quy tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong ngữ cảnh. Không tự suy diễn hoặc bịa đặt thông tin.
2. Nếu ngữ cảnh không chứa đủ thông tin để trả lời, hãy trả lời:
"Dựa trên tài liệu được cung cấp, tôi không tìm thấy thông tin đầy đủ để trả lời câu hỏi này."
3. Khi trả lời, hãy dẫn nguồn nếu metadata có sẵn, bao gồm tên file, tiêu đề mục và số trang.
4. Phân biệt rõ thông tin được tài liệu nêu trực tiếp với thông tin không được đề cập.
5. Với các câu hỏi về số liệu, tên mô hình, epoch, chỉ số đánh giá hoặc cấu hình kỹ thuật, phải giữ chính xác giá trị được nêu trong ngữ cảnh. Không tự thay thế bằng giá trị khác.
6. Trả lời bằng tiếng Việt chuẩn mực, mạch lạc và định dạng rõ ràng."""


DEFAULT_USER_TEMPLATE = """Dưới đây là các đoạn ngữ cảnh trích xuất từ tài liệu:

{context}

--------------------------------------------------------------------------------
Câu hỏi của người dùng:
{query}

Hãy trả lời dựa trên các ngữ cảnh được cung cấp. Nêu nguồn tham khảo từ tên file, đường dẫn mục và số trang khi có thông tin."""


class PromptBuilder:
    """Định dạng context và xây dựng prompt cho mô hình LLM."""

    def __init__(
        self,
        template_path: Optional[Union[str, Path]] = None,
    ):
        cfg = get_config()

        self.template_path = (
            Path(template_path)
            if template_path
            else cfg.prompts.qa_path
        )

        self.system_prompt, self.user_template = self._load_template()

    def _load_template(self) -> Tuple[str, str]:
        """Nạp prompt template từ file YAML."""
        if self.template_path.exists():
            try:
                with open(
                    self.template_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    data = yaml.safe_load(file) or {}

                system_prompt = data.get(
                    "system_prompt",
                    DEFAULT_SYSTEM_PROMPT,
                ).strip()

                user_template = data.get(
                    "user_template",
                    DEFAULT_USER_TEMPLATE,
                ).strip()

                logger.info(
                    "Đã nạp prompt template từ: %s",
                    self.template_path.name,
                )

                return system_prompt, user_template

            except Exception as exc:
                logger.warning(
                    "Không thể đọc prompt template %s: %s. "
                    "Sử dụng template mặc định.",
                    self.template_path,
                    exc,
                )

        return DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE

    @staticmethod
    def _get_value(
        item: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        """Đọc dữ liệu từ dictionary hoặc object."""
        if isinstance(item, dict):
            return item.get(key, default)

        return getattr(item, key, default)

    @classmethod
    def _get_metadata_value(
        cls,
        chunk: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Đọc metadata từ cấu trúc chunk.metadata hoặc field trực tiếp.

        Tương thích với:
        - FormattedChunk từ Generator.
        - RetrievedChunk hiện có.
        - Dictionary chứa text và metadata.
        """
        metadata = cls._get_value(chunk, "metadata", {}) or {}

        value = cls._get_value(metadata, key)

        if value is not None:
            return value

        value = cls._get_value(chunk, key)

        return value if value is not None else default

    @classmethod
    def _get_text(cls, chunk: Any) -> str:
        """Lấy nội dung chunk từ các kiểu dữ liệu phổ biến."""
        for field in ("text", "content", "page_content", "chunk_text"):
            content = cls._get_value(chunk, field)

            if isinstance(content, str):
                return content.strip()

        return ""

    @classmethod
    def _get_breadcrumb(cls, chunk: Any) -> str:
        """Lấy breadcrumb hoặc tạo fallback từ section_path."""
        breadcrumb = cls._get_metadata_value(chunk, "breadcrumb")

        if isinstance(breadcrumb, str) and breadcrumb.strip():
            return breadcrumb.strip()

        section_path = cls._get_metadata_value(
            chunk,
            "section_path",
        )

        if isinstance(section_path, (list, tuple)) and section_path:
            return " > ".join(
                str(part).strip()
                for part in section_path
                if str(part).strip()
            )

        section_title = cls._get_metadata_value(
            chunk,
            "section_title",
        )

        if section_title:
            return str(section_title).strip()

        return "Không xác định"

    @classmethod
    def _get_page(cls, chunk: Any) -> Optional[Any]:
        """
        Lấy trang từ metadata đã format hoặc metadata gốc.

        Không tự suy đoán số trang nếu dữ liệu nguồn không cung cấp.
        """
        # Cấu trúc rút gọn từ Generator.
        page = cls._get_metadata_value(chunk, "page")

        if page is not None:
            return page

        # Cấu trúc metadata gốc.
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

        return cls._get_metadata_value(chunk, "page_number")

    @classmethod
    def _format_single_chunk(
        cls,
        chunk: Any,
        index: int,
    ) -> Optional[str]:
        """Format một chunk với đúng 3 metadata cần thiết."""
        content = cls._get_text(chunk)

        if not content:
            return None

        file_name = cls._get_metadata_value(
            chunk,
            "file_name",
            "Không xác định",
        )

        breadcrumb = cls._get_breadcrumb(chunk)
        page = cls._get_page(chunk)

        if file_name is None or not str(file_name).strip():
            file_name = "Không xác định"

        page_text = (
            str(page)
            if page is not None and str(page).strip()
            else "Không xác định"
        )

        return (
            f"[Đoạn ngữ cảnh {index}]\n"
            f"- Tên file: {file_name}\n"
            f"- Đường dẫn mục (breadcrumb): {breadcrumb}\n"
            f"- Trang: {page_text}\n"
            f"Nội dung:\n{content}"
        )

    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format các chunks thành context cho LLM.

        Chỉ đưa vào:
        - text
        - breadcrumb
        - file_name
        - page

        Việc loại bỏ chunks trùng lặp được thực hiện trong Generator.
        """
        if not chunks:
            return "Không tìm thấy đoạn trích dẫn ngữ cảnh nào."

        context_parts: List[str] = []
        empty_count = 0

        for index, chunk in enumerate(chunks, start=1):
            formatted_chunk = self._format_single_chunk(
                chunk=chunk,
                index=index,
            )

            if formatted_chunk is None:
                empty_count += 1
                continue

            context_parts.append(formatted_chunk)

        if empty_count:
            logger.warning(
                "Bỏ qua %d chunks rỗng khi format context.",
                empty_count,
            )

        if not context_parts:
            return "Không tìm thấy đoạn trích dẫn ngữ cảnh nào."

        return "\n\n---\n\n".join(context_parts)

    def build_prompt(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> Tuple[str, str]:
        """Tạo cặp system_prompt và user_prompt."""
        context_str = self.format_context(chunks)

        user_prompt = self.user_template.format(
            context=context_str,
            query=query.strip(),
        )

        return self.system_prompt, user_prompt

    def build_messages(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> List[Dict[str, str]]:
        """Tạo messages theo định dạng chat OpenAI/Groq."""
        system_prompt, user_prompt = self.build_prompt(
            query,
            chunks,
        )

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
