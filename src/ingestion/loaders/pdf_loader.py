import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pdfplumber

from src.config import get_config
from src.ingestion.loaders.base import BaseLoader
from src.ingestion.loaders.block_adapter import documents_from_blocks
from src.logger import get_logger
from src.schemas.blocks import (
    BlockType,
    BoundingBox,
    DocumentRegion,
    ParsedBlock,
    TableData,
)
from src.schemas.document import Document

logger = get_logger("pdf_loader")

# Tỷ lệ font size so với median toàn trang để suy luận heading
HEADING_FONT_RATIO = 1.15   # To hơn median 15% -> heading
TABLE_CAPTION_PATTERN = re.compile(r"^(?:bảng|table)\s*(?:số\s*)?\d+(?:[.-]\d+)*\b", re.IGNORECASE)
CHAPTER_HEADING_PATTERN = re.compile(r"^(?:chương|chapter)\s+\d+\b", re.IGNORECASE)
NUMBERED_HEADING_PATTERN = re.compile(r"^\d+(?:\.\d+)*(?:[.)])?\s+\S")
LETTERED_HEADING_PATTERN = re.compile(r"^[A-ZĐ]\.\s+\S")
ROMAN_HEADING_PATTERN = re.compile(r"^[IVXLCDM]+\.\s+\S", re.IGNORECASE)
LIST_ITEM_PATTERN = re.compile(
    r"^(?:[-–—•*]\s*|thứ\s+(?:nhất|hai|ba|tư|năm|sáu|bảy|tám|chín|mười)\b|"
    r"(?:một|hai|ba|bốn|năm)\s+là\b)",
    re.IGNORECASE,
)


def _classify_line(
    chars: list,
    median_size: float,
    heading_font_ratio: float = HEADING_FONT_RATIO,
) -> Tuple[bool, str]:
    """Phân loại heading theo cấu trúc, có hỗ trợ kiểu chữ lớn.
    
    Returns:
        (is_heading: bool, text: str)
    """
    if not chars:
        return False, ""

    sizes = [c.get("size", 0) for c in chars if c.get("size")]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    text = pdfplumber.utils.extract_text(chars, x_tolerance=3).strip()

    if not text:
        return False, ""

    is_big = avg_size >= median_size * heading_font_ratio
    has_bold_style = any(
        any(
            style in c.get("fontname", "").lower()
            for style in ("bold", "black", "heavy", "demi")
        )
        for c in chars
    )
    short_enough = len(text) <= 200  # Heading thường ngắn
    is_list_item = bool(LIST_ITEM_PATTERN.match(text))
    is_structured_heading = bool(
        CHAPTER_HEADING_PATTERN.match(text)
        or NUMBERED_HEADING_PATTERN.match(text)
        or LETTERED_HEADING_PATTERN.match(text)
        or ROMAN_HEADING_PATTERN.match(text)
    )

    # Chữ đậm chỉ hỗ trợ heading có cấu trúc, không tự biến nội dung thành heading.
    is_heading = short_enough and not is_list_item and (
        is_big or (is_structured_heading and has_bold_style)
    )
    return is_heading, text


def _extract_heading_level(text: str) -> int:
    """Suy luận cấp heading từ chương, số thứ tự hoặc nhãn mục chữ cái."""
    if CHAPTER_HEADING_PATTERN.match(text.strip()):
        return 1

    match = re.match(r"^(\d+(?:\.\d+)*)\s*\.?\s", text.strip())
    if match:
        return min(len(match.group(1).split(".")), 4)

    if LETTERED_HEADING_PATTERN.match(text.strip()) or ROMAN_HEADING_PATTERN.match(
        text.strip()
    ):
        return 2

    return 1


def _is_table_caption(text: str) -> bool:
    caption = text.strip().lstrip("*#- ").strip()
    return bool(TABLE_CAPTION_PATTERN.match(caption))


class PDFLoader(BaseLoader):
    """Loader xử lý PDF với pdfplumber: phát hiện heading theo font size,
    gộp bảng vào đúng section đang active, gắn metadata breadcrumb."""

    def __init__(
        self,
        clean_content: bool = True,
        heading_font_ratio: float = HEADING_FONT_RATIO,
        ocr_enabled: Optional[bool] = None,
    ):
        self.clean_content = clean_content
        self.heading_font_ratio = heading_font_ratio
        self.ocr_config = get_config().data.pdf_ocr
        self.ocr_enabled = self.ocr_config.enabled if ocr_enabled is None else ocr_enabled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_median_font_size(self, page) -> float:
        """Tính median font size của toàn trang."""
        sizes = [
            c.get("size", 0)
            for c in page.chars
            if c.get("size", 0) > 0
        ]
        if not sizes:
            return 12.0
        sizes.sort()
        mid = len(sizes) // 2
        return sizes[mid]

    def _page_to_blocks(self, page) -> List[dict]:
        """Chuyển trang PDF thành block text/table trung gian có cấu trúc.
        
        Chiến lược:
        1. Trích xuất tất cả bảng từ trang -> giữ header/rows/caption -> đánh dấu bbox.
        2. Với từng dòng text còn lại (ngoài vùng bảng), phân loại heading/content.
        3. Kết quả là list block theo thứ tự xuất hiện dựa trên tọa độ y trên trang.
        """
        blocks: List[dict] = []
        table_bboxes: List[tuple] = []

        # 1. Xử lý bảng
        for tbl_settings in [
            {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
            {"vertical_strategy": "text", "horizontal_strategy": "text"},
        ]:
            try:
                found_tables = page.find_tables(tbl_settings)
            except Exception as exc:
                logger.warning(
                    "Không extract được bảng ở trang %s: %s",
                    page.page_number if hasattr(page, "page_number") else "?",
                    exc,
                )

            table_block_count = len(blocks)
            for found_tbl in found_tables:
                try:
                    rows = found_tbl.extract()
                    if not rows:
                        continue
                    bbox = found_tbl.bbox
                    table = TableData.from_matrix(rows)
                    if table.columns:
                        table_bboxes.append(bbox)
                        avg_y = (bbox[1] + bbox[3]) / 2
                        blocks.append({"type": "table", "y": avg_y, "table": table, "bbox": bbox})
                except Exception:
                    continue

            if len(blocks) > table_block_count:
                break

        # 2. Xử lý text theo dòng, bỏ qua vùng bảng
        median_size = self._get_median_font_size(page)

        def _in_table(char_bbox: tuple) -> bool:
            cx = (char_bbox[0] + char_bbox[2]) / 2
            cy = (char_bbox[1] + char_bbox[3]) / 2
            for bx0, by0, bx1, by1 in table_bboxes:
                if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                    return True
            return False

        # Gom char thành dòng theo tọa độ y (làm tròn 2 chữ số)
        line_dict: dict = {}
        for char in page.chars:
            if _in_table((char["x0"], char["top"], char["x1"], char["bottom"])):
                continue
            y_key = round(char.get("top", 0), 1)
            if y_key not in line_dict:
                line_dict[y_key] = []
            line_dict[y_key].append(char)

        for y_key in sorted(line_dict.keys()):
            chars = sorted(line_dict[y_key], key=lambda c: c.get("x0", 0))
            is_heading, text = _classify_line(
                chars,
                median_size,
                heading_font_ratio=self.heading_font_ratio,
            )
            if not text:
                continue
            blocks.append({
                "type": "heading" if is_heading else "text",
                "y": y_key,
                "content": text,
                "level": _extract_heading_level(text) if is_heading else 0,
            })

        # Sắp xếp theo thứ tự xuất hiện trên trang
        blocks.sort(key=lambda b: b["y"])

        # Caption in đậm thường bị nhận diện là heading; gắn nó vào bảng kế tiếp.
        merged_blocks: List[dict] = []
        for block in blocks:
            if (
                block["type"] == "table"
                and merged_blocks
                and merged_blocks[-1]["type"] in {"heading", "text"}
                and _is_table_caption(merged_blocks[-1]["content"])
            ):
                caption = merged_blocks.pop()
                block["table"].caption = caption["content"]
            elif (
                block["type"] in {"heading", "text"}
                and merged_blocks
                and merged_blocks[-1]["type"] == "table"
                and _is_table_caption(block["content"])
            ):
                table = merged_blocks[-1]
                table["table"].caption = block["content"]
                continue
            merged_blocks.append(block)
        return merged_blocks

    def _ocr_page(self, file_path: Path, page_index: int) -> str:
        """Render one page and recognize text with the configured Tesseract languages."""
        try:
            import pymupdf
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu thư viện OCR. Cài đặt các gói trong requirements.txt rồi thử lại."
            ) from exc

        if self.ocr_config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.ocr_config.tesseract_cmd

        try:
            available_languages = set(pytesseract.get_languages(config=""))
        except Exception as exc:
            raise RuntimeError(
                "Không chạy được Tesseract. Hãy cài Tesseract OCR và cấu hình tesseract_cmd nếu cần."
            ) from exc

        requested_languages = [
            language.strip()
            for language in self.ocr_config.languages.split("+")
            if language.strip()
        ]
        selected_languages = [
            language for language in requested_languages if language in available_languages
        ]
        missing_languages = [
            language for language in requested_languages if language not in available_languages
        ]
        if missing_languages:
            logger.warning(
                f"Tesseract thiếu language data {', '.join(missing_languages)}. "
                f"Đang dùng: {'+'.join(selected_languages) or 'eng nếu khả dụng'}. "
                "Cài traineddata tương ứng để OCR chính xác hơn."
            )
        if not selected_languages and "eng" in available_languages:
            selected_languages = ["eng"]
        if not selected_languages:
            raise RuntimeError(
                "Không có Tesseract language data phù hợp. "
                "Hãy cài ít nhất ngôn ngữ eng hoặc cấu hình data.pdf_ocr.languages."
            )

        try:
            with pymupdf.open(str(file_path)) as pdf:
                page = pdf[page_index]
                pixmap = page.get_pixmap(dpi=self.ocr_config.dpi, alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            return pytesseract.image_to_string(
                image,
                lang="+".join(selected_languages),
                config="--psm 3",
            ).strip()
        except Exception as exc:
            raise RuntimeError(f"OCR thất bại ở trang {page_index + 1}: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_blocks(self, file_path: Union[str, Path]) -> List[ParsedBlock]:
        """Parse a PDF into normalized blocks without flattening tables."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file PDF: {path}")

        logger.info(f"Đang đọc file PDF bằng pdfplumber: {path.name}")
        parsed_blocks: List[ParsedBlock] = []

        # Stack tiêu đề đang active: [(level, title), ...]
        heading_stack: List[Tuple[int, str]] = []
        total_pages = 0

        try:
            with pdfplumber.open(str(path)) as pdf:
                total_pages = len(pdf.pages)

                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    blocks = self._page_to_blocks(page)
                    page_ocr_used = False
                    native_char_count = len(page.chars)
                    has_native_table = any(block["type"] == "table" for block in blocks)

                    if (
                        self.ocr_enabled
                        and native_char_count < self.ocr_config.min_text_chars
                        and not has_native_table
                    ):
                        try:
                            ocr_text = self._ocr_page(path, page_idx)
                        except RuntimeError as exc:
                            if native_char_count == 0:
                                raise
                            logger.warning(
                                f"Không thể OCR trang {page_num} của {path.name}; "
                                f"giữ lại text native: {exc}"
                            )
                        else:
                            if ocr_text:
                                blocks = [{"type": "text", "y": 0, "content": ocr_text}]
                                page_ocr_used = True
                                logger.info(f"Đã OCR trang {page_num} của {path.name}")

                    for block in blocks:
                        if block["type"] == "heading":
                            # Cập nhật heading stack
                            lvl = block["level"]
                            # Bỏ bất kỳ heading có cấp >= level (giữ tiêu đề cha)
                            heading_stack = [(l, t) for l, t in heading_stack if l < lvl]
                            heading_stack.append((lvl, block["content"]))

                        section_path = [title for _, title in heading_stack]
                        breadcrumb = " > ".join(section_path)
                        chapter = section_path[0] if section_path else ""
                        current_section = section_path[-1] if section_path else f"Trang {page_num}"
                        meta: dict = {
                            "source": str(path),
                            "file_name": path.name,
                            "file_type": "pdf",
                            "total_pages": total_pages,
                            "chapter": chapter,
                            "section_title": current_section,
                            "breadcrumb": breadcrumb,
                        }
                        if page_ocr_used:
                            meta["ocr_used"] = True

                        block_type = (
                            BlockType.HEADING if block["type"] == "heading"
                            else BlockType.TABLE if block["type"] == "table"
                            else BlockType.PARAGRAPH
                        )
                        bbox = None
                        if block.get("bbox"):
                            x0, top, x1, bottom = block["bbox"]
                            bbox = BoundingBox(x0=x0, top=top, x1=x1, bottom=bottom)
                        parsed_blocks.append(
                            ParsedBlock(
                                block_index=len(parsed_blocks),
                                block_type=block_type,
                                text=block.get("content"),
                                heading_level=block.get("level") if block_type == BlockType.HEADING else None,
                                section_path=section_path,
                                section_id=ParsedBlock.section_identifier(str(path), section_path),
                                region=DocumentRegion.BODY,
                                page_number=page_num,
                                bbox=bbox,
                                table=block.get("table"),
                                metadata=meta,
                            )
                        )

            logger.info(
                f"Đã trích xuất {len(parsed_blocks)} blocks từ {total_pages} trang của {path.name}"
            )
            return parsed_blocks

        except Exception as e:
            logger.error(f"Lỗi khi đọc file PDF {path.name}: {e}")
            raise

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """Compatibility API returning the legacy ``List[Document]`` contract."""
        return documents_from_blocks(
            self.load_blocks(file_path),
            clean_content=self.clean_content,
        )
