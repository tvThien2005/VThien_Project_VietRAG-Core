import re
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple, Union

import docx
from docx.oxml.ns import qn
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from src.ingestion.loaders.base import BaseLoader
from src.ingestion.loaders.block_adapter import documents_from_blocks
from src.logger import get_logger
from src.schemas.blocks import (
    BlockType,
    DocumentRegion,
    ParsedBlock,
    TableCell,
    TableData,
)
from src.schemas.document import Document

logger = get_logger("docx_loader")

_CHAPTER_PATTERN = re.compile(r"^(?:chương|chapter)\s+\d+\b", re.IGNORECASE)
_NUMBERED_HEADING_PATTERN = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+\S")
_SINGLE_NUMBER_HEADING_PATTERN = re.compile(r"^(\d+)\.\s+\S")
_LIST_ITEM_PATTERN = re.compile(r"^(?:[-–—•*]\s+|\(?\d+[)]\s+|\(?[a-zđA-ZĐ][)]\s+)")
_CAPTION_PATTERN = re.compile(r"^(?:bảng|table)\s*(?:số\s*)?\d*(?:[.\-]\d+)*\b", re.IGNORECASE)
_TOC_MARKER_PATTERN = re.compile(r"^(?:mục\s*lục|table\s+of\s+contents)\b", re.IGNORECASE)
_FIGURES_MARKER_PATTERN = re.compile(r"^(?:danh\s*mục\s*hình|list\s+of\s+figures)\b", re.IGNORECASE)
_TABLES_MARKER_PATTERN = re.compile(r"^(?:danh\s*mục\s*(?:các\s*)?bảng|list\s+of\s+tables)\b", re.IGNORECASE)
_REFERENCES_PATTERN = re.compile(r"^(?:tài\s*liệu\s*tham\s*khảo|tham\s*khảo|references?|bibliography)\b", re.IGNORECASE)
_TOC_REGIONS = {DocumentRegion.TABLE_OF_CONTENTS, DocumentRegion.LIST_OF_FIGURES, DocumentRegion.LIST_OF_TABLES}


def _iter_block_items(doc: docx.Document) -> Iterable[Union[Paragraph, Table]]:
    """Duyệt qua Paragraph VÀ Table theo đúng thứ tự xuất hiện trong file DOCX.
    
    python-docx không cung cấp API duyệt hỗn hợp paragraph + table theo thứ tự,
    nên ta phải đọc trực tiếp từ XML body để giữ đúng vị trí.
    """
    parent_elm = doc.element.body
    for child in parent_elm.iterchildren():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            yield Paragraph(child, doc)
        elif tag == "tbl":
            yield Table(child, doc)


def _element_value(element, name: str, default: Optional[str] = None) -> Optional[str]:
    return element.get(qn(name), default) if element is not None else default


def _paragraph_has_toc_field(paragraph: Paragraph) -> bool:
    xml = paragraph._p.xml.upper()
    return "TOC" in xml and ("INSTRTEXT" in xml or "FLDSIMPLE" in xml)


def _toc_field_state(paragraph: Paragraph, active: bool) -> Tuple[bool, bool]:
    """Return whether this paragraph belongs to a TOC field and its next state."""
    xml = paragraph._p.xml.upper()
    has_instruction = "TOC" in xml and ("INSTRTEXT" in xml or "FLDSIMPLE" in xml)
    has_simple_field = "FLDSIMPLE" in xml and has_instruction
    ends_field = bool(re.search(r'FLDCHAR[^>]*FLDCHARTYPE=["\']END["\']', xml))
    in_toc_field = active or (has_instruction and not has_simple_field)
    belongs_to_toc = has_instruction or in_toc_field
    if ends_field:
        in_toc_field = False
    elif has_instruction and not has_simple_field:
        in_toc_field = True
    return belongs_to_toc, in_toc_field


def _is_toc_style(paragraph: Paragraph) -> bool:
    style = paragraph.style
    values = (
        (style.name if style else "") or "",
        (getattr(style, "style_id", "") if style else "") or "",
    )
    return any(
        re.match(
            r"^(?:toc(?:\s*(?:heading|\d+))?|table\s+of\s+contents(?:\s*\d+)?|mục\s*lục(?:\s*\d+)?)$",
            value.strip(),
            re.IGNORECASE,
        )
        for value in values
    )


def _style_heading_level(paragraph: Paragraph) -> Optional[int]:
    style = paragraph.style
    for value in ((style.name if style else "") or "", (getattr(style, "style_id", "") if style else "") or ""):
        match = re.search(r"(?:heading|tiêu\s*đề|tieu\s*de)\s*([1-6])\b", value, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _outline_heading_level(paragraph: Paragraph) -> Optional[int]:
    p_pr = paragraph._p.pPr
    value = _element_value(p_pr.find(qn("w:outlineLvl")) if p_pr is not None else None, "w:val")
    try:
        level = int(value) + 1 if value is not None else 0
    except ValueError:
        return None
    return level if 1 <= level <= 6 else None


def _is_direct_heading(paragraph: Paragraph) -> bool:
    text = paragraph.text.strip()
    if (
        not text
        or len(text) > 140
        or len(text.split()) > 16
        or text.endswith((".", "?", "!", ";", ":"))
        or text.isupper()
        or _is_list_paragraph(paragraph)
    ):
        return False
    if _LIST_ITEM_PATTERN.match(text) or _CAPTION_PATTERN.match(text):
        return False
    runs = [run for run in paragraph.runs if run.text.strip()]
    sizes = [float(run.font.size.pt) for run in runs if run.font.size is not None]
    normal = paragraph.part.document.styles["Normal"].font.size
    normal_size = float(normal.pt) if normal is not None else 11.0
    return bool(runs and sizes) and sum(run.bold is True for run in runs) / len(runs) >= 0.75 and max(sizes) >= normal_size * 1.15


def _is_list_paragraph(paragraph: Paragraph) -> bool:
    style = paragraph.style
    style_names = (
        (style.name if style else "") or "",
        (getattr(style, "style_id", "") if style else "") or "",
    )
    if any(re.search(r"(?:^|[\s_-])(?:list|bullet)(?:$|[\s_-]|\d)", name, re.IGNORECASE) for name in style_names):
        return True
    p_pr = paragraph._p.pPr
    return p_pr is not None and p_pr.find(qn("w:numPr")) is not None


def _heading_shaped_text(text: str) -> bool:
    return bool(
        text
        and len(text) <= 140
        and len(text.split()) <= 16
        and not text.endswith((".", "?", "!", ";", ":"))
    )


def _numbered_heading_level(text: str, direct_heading: bool) -> Optional[int]:
    if _CHAPTER_PATTERN.match(text):
        return 1 if _heading_shaped_text(text) else None
    if not _heading_shaped_text(text):
        return None
    match = _NUMBERED_HEADING_PATTERN.match(text)
    if match:
        return min(len(match.group(1).split(".")), 6)
    return 1 if direct_heading and _SINGLE_NUMBER_HEADING_PATTERN.match(text) else None


def _heading_level(paragraph: Paragraph) -> Optional[int]:
    text = paragraph.text.strip()
    if (
        not text
        or _paragraph_has_toc_field(paragraph)
        or _is_toc_style(paragraph)
        or _CAPTION_PATTERN.match(text)
        or _LIST_ITEM_PATTERN.match(text)
    ):
        return None
    style_level = _style_heading_level(paragraph)
    outline_level = _outline_heading_level(paragraph)
    if style_level is not None:
        return style_level
    if outline_level is not None:
        return outline_level
    if _is_list_paragraph(paragraph):
        return None
    direct_heading = _is_direct_heading(paragraph)
    return _numbered_heading_level(text, direct_heading) or (1 if direct_heading else None)


def _region_marker(text: str) -> Optional[DocumentRegion]:
    if _TOC_MARKER_PATTERN.match(text):
        return DocumentRegion.TABLE_OF_CONTENTS
    if _FIGURES_MARKER_PATTERN.match(text):
        return DocumentRegion.LIST_OF_FIGURES
    if _TABLES_MARKER_PATTERN.match(text):
        return DocumentRegion.LIST_OF_TABLES
    if _REFERENCES_PATTERN.match(text):
        return DocumentRegion.REFERENCES
    return None


def _is_main_content_start(text: str, level: Optional[int]) -> bool:
    return bool(level and text and not _region_marker(text))


def _header_row_count(table: Table) -> int:
    count = 0
    for tr in table._tbl.tr_lst:
        tr_pr = tr.trPr
        marker = tr_pr.find(qn("w:tblHeader")) if tr_pr is not None else None
        if marker is None or _element_value(marker, "w:val", "1") in {"0", "false", "False", "off"}:
            break
        count += 1
    return count


def _table_to_data(table: Table, caption: Optional[str] = None) -> TableData:
    """Keep logical cells and OOXML merge spans; never assume row zero is a header."""
    raw_rows = []
    maximum_columns = 0
    for row_index, tr in enumerate(table._tbl.tr_lst):
        column_index = 0
        expanded, logical_cells = [], []
        for tc in tr.tc_lst:
            tc_pr = tc.tcPr
            grid_span = tc_pr.find(qn("w:gridSpan")) if tc_pr is not None else None
            try:
                column_span = int(_element_value(grid_span, "w:val", "1") or "1")
            except ValueError:
                column_span = 1
            v_merge = tc_pr.find(qn("w:vMerge")) if tc_pr is not None else None
            vertical_merge = _element_value(v_merge, "w:val", "continue") if v_merge is not None else None
            text = _Cell(tc, table).text.strip().replace("\n", " ")
            logical_cells.append(TableCell(row_index=row_index, column_index=column_index, text=text, column_span=column_span, vertical_merge=vertical_merge))
            expanded.extend([text] + [""] * (column_span - 1))
            column_index += column_span
        maximum_columns = max(maximum_columns, column_index)
        raw_rows.append((expanded, logical_cells))
    for row_index, (_, cells) in enumerate(raw_rows):
        for cell in cells:
            if cell.vertical_merge != "restart":
                continue
            span = 1
            for _, following_cells in raw_rows[row_index + 1:]:
                if not any(next_cell.column_index == cell.column_index and next_cell.vertical_merge == "continue" for next_cell in following_cells):
                    break
                span += 1
            cell.row_span = span
    matrix = [row + [""] * (maximum_columns - len(row)) for row, _ in raw_rows]
    return TableData.from_matrix(matrix, caption=caption, header_row_count=_header_row_count(table), cells=[cells for _, cells in raw_rows])


def _is_meaningful_table(table: TableData) -> bool:
    return any(re.search(r"\w", cell.text, re.UNICODE) for row in table.cells for cell in row)


class DocxLoader(BaseLoader):
    """DOCX loader exposing normalized blocks and a legacy Document adapter."""

    def __init__(
        self,
        clean_content: bool = True,
        index_regions: Optional[Set[Union[DocumentRegion, str]]] = None,
    ):
        self.clean_content = clean_content
        self.index_regions = self._normalize_index_regions(index_regions)

    @staticmethod
    def _normalize_index_regions(
        regions: Optional[Set[Union[DocumentRegion, str]]],
    ) -> Optional[Set[DocumentRegion]]:
        if regions is None:
            return None
        return {region if isinstance(region, DocumentRegion) else DocumentRegion(region) for region in regions}

    def should_index_block(self, block: ParsedBlock) -> bool:
        return self.index_regions is None or block.region in self.index_regions

    def _get_heading_level(self, paragraph: Paragraph) -> Optional[int]:
        """Compatibility wrapper for callers of the former private method."""
        return _heading_level(paragraph)

    def load_blocks(
        self,
        file_path: Union[str, Path],
        *,
        indexable_only: bool = False,
    ) -> List[ParsedBlock]:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file DOCX: {path}")

        logger.info(f"Đang đọc file DOCX: {path.name}")
        document = docx.Document(str(path))
        parsed_blocks: List[ParsedBlock] = []
        heading_stack: List[Tuple[int, str]] = []
        section_index = 0
        region = DocumentRegion.COVER
        last_caption: Optional[ParsedBlock] = None
        last_table_index: Optional[int] = None
        toc_field_active = False

        def metadata(section_path: List[str]) -> dict:
            return {
                "source": str(path),
                "file_name": path.name,
                "file_type": "docx",
                "section_index": section_index,
                "section_title": section_path[-1] if section_path else "Mở đầu",
                "chapter": section_path[0] if section_path else "",
                "breadcrumb": " > ".join(section_path),
            }

        def append_block(
            block_type: BlockType,
            *,
            text: Optional[str] = None,
            level: Optional[int] = None,
            table: Optional[TableData] = None,
            extra_metadata: Optional[dict] = None,
        ) -> ParsedBlock:
            section_path = [title for _, title in heading_stack]
            block_metadata = metadata(section_path)
            if extra_metadata:
                block_metadata.update(extra_metadata)
            parsed = ParsedBlock(
                block_index=len(parsed_blocks),
                block_type=block_type,
                text=text,
                heading_level=level,
                section_path=section_path,
                section_id=ParsedBlock.section_identifier(str(path), section_path),
                region=region,
                table=table,
                metadata=block_metadata,
            )
            parsed_blocks.append(parsed)
            return parsed

        for item in _iter_block_items(document):
            if isinstance(item, Paragraph):
                text = item.text.strip()
                marker = _region_marker(text) if text else None
                belongs_to_toc_field, toc_field_active = _toc_field_state(item, toc_field_active)
                toc_style = _is_toc_style(item)
                if (
                    marker is not None
                    and toc_style
                    and re.search(r"(?:\t|\.{2,})\s*\d+\s*$", text)
                ):
                    marker = None
                if marker is not None:
                    region = marker
                elif belongs_to_toc_field or toc_style:
                    region = DocumentRegion.TABLE_OF_CONTENTS
                if not text:
                    continue

                level = self._get_heading_level(item)
                if region in _TOC_REGIONS and marker is None:
                    if not toc_style and not belongs_to_toc_field and _is_main_content_start(text, level):
                        region = DocumentRegion.BODY
                    else:
                        level = None
                elif region == DocumentRegion.COVER and marker is None and _is_main_content_start(text, level):
                    region = DocumentRegion.BODY

                if level is not None and region not in _TOC_REGIONS:
                    heading_stack[:] = [(old_level, title) for old_level, title in heading_stack if old_level < level]
                    heading_stack.append((level, text))
                    if parsed_blocks:
                        section_index += 1
                    append_block(BlockType.HEADING, text=text, level=level)
                    last_caption = None
                    last_table_index = None
                elif _CAPTION_PATTERN.match(text):
                    caption_block = append_block(BlockType.CAPTION, text=text)
                    if last_table_index is not None:
                        table_block = parsed_blocks[last_table_index]
                        table_metadata = dict(table_block.metadata)
                        table_metadata.pop("caption_missing", None)
                        table_metadata["caption_block_id"] = caption_block.block_id
                        caption_block = caption_block.model_copy(
                            update={
                                "metadata": {
                                    **caption_block.metadata,
                                    "table_block_id": table_block.block_id,
                                }
                            }
                        )
                        table_data = table_block.table.model_copy(
                            update={"caption": text}
                        )
                        parsed_blocks[last_table_index] = table_block.model_copy(
                            update={"table": table_data, "metadata": table_metadata}
                        )
                        parsed_blocks[-1] = caption_block
                        last_table_index = None
                        last_caption = None
                    else:
                        last_caption = caption_block
                else:
                    append_block(BlockType.PARAGRAPH, text=text)
                    last_caption = None
                    last_table_index = None

            elif isinstance(item, Table):
                caption = last_caption.text if last_caption else None
                table = _table_to_data(item, caption=caption)
                if not _is_meaningful_table(table):
                    last_caption = None
                    last_table_index = None
                    continue
                extra = {"caption_block_id": last_caption.block_id} if last_caption else {"caption_missing": True}
                table_block = append_block(BlockType.TABLE, table=table, extra_metadata=extra)
                last_table_index = None if last_caption else table_block.block_index
                last_caption = None

        if indexable_only:
            parsed_blocks = [block for block in parsed_blocks if self.should_index_block(block)]
            parsed_blocks = [block.model_copy(update={"block_index": index}) for index, block in enumerate(parsed_blocks)]

        logger.info(f"Đã trích xuất {len(parsed_blocks)} blocks từ {path.name}")
        return parsed_blocks

    def load_indexable_blocks(self, file_path: Union[str, Path]) -> List[ParsedBlock]:
        return self.load_blocks(file_path, indexable_only=True)

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        return documents_from_blocks(
            self.load_blocks(file_path, indexable_only=self.index_regions is not None),
            clean_content=self.clean_content,
        )
