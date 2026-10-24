"""Normalized intermediate representation for parsed document content.

The classes in this module deliberately keep parsing output separate from the
legacy ``Document`` text contract.  Loaders can expose lossless structural
blocks while existing ingestion code continues to consume ``Document``.
"""

from __future__ import annotations

from enum import Enum
from hashlib import sha1
from typing import Any, Dict, List, Optional, Sequence
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator


class BlockType(str, Enum):
    """Supported structural block categories across all loaders."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    EQUATION = "equation"
    CAPTION = "caption"
    CODE = "code"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_BREAK = "page_break"


class DocumentRegion(str, Enum):
    """Coarse document region, kept unknown when the parser has no evidence."""

    UNKNOWN = "unknown"
    COVER = "cover"
    FRONT_MATTER = "front_matter"
    TABLE_OF_CONTENTS = "table_of_contents"
    LIST_OF_FIGURES = "list_of_figures"
    LIST_OF_TABLES = "list_of_tables"
    BODY = "body"
    APPENDIX = "appendix"
    REFERENCES = "references"
    HEADER = "header"
    FOOTER = "footer"


class BoundingBox(BaseModel):
    """Coordinates in the source page coordinate system when available."""

    x0: float
    top: float
    x1: float
    bottom: float

    @model_validator(mode="after")
    def validate_order(self) -> "BoundingBox":
        if self.x1 < self.x0:
            raise ValueError("bbox.x1 must be greater than or equal to bbox.x0")
        if self.bottom < self.top:
            raise ValueError("bbox.bottom must be greater than or equal to bbox.top")
        return self


class TableColumn(BaseModel):
    """A logical table column. ``name`` may be empty for an unlabeled column."""

    index: int = Field(ge=0)
    name: str = ""


class TableCell(BaseModel):
    """A source cell with merge span information when the extractor exposes it."""

    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    text: str = ""
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    vertical_merge: Optional[str] = None


class TableData(BaseModel):
    """Lossless-enough tabular representation used before chunking.

    The header is kept separately from data rows.  Cells are strings because
    document extractors do not provide reliable typed values at this stage.
    Rows may be shorter than the column count for merged/ragged source tables;
    they may never be wider than the declared schema.
    """

    table_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    caption: Optional[str] = None
    columns: List[TableColumn] = Field(default_factory=list)
    header: List[str] = Field(default_factory=list)
    header_rows: List[List[str]] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    cells: List[List[TableCell]] = Field(default_factory=list)

    @field_validator("caption")
    @classmethod
    def normalize_caption(cls, value: Optional[str]) -> Optional[str]:
        value = value.strip() if value else None
        return value or None

    @field_validator("header")
    @classmethod
    def normalize_header(cls, value: List[str]) -> List[str]:
        return [str(cell or "").strip() for cell in value]

    @field_validator("rows")
    @classmethod
    def normalize_rows(cls, value: List[List[str]]) -> List[List[str]]:
        return [[str(cell or "").strip() for cell in row] for row in value]

    @field_validator("header_rows")
    @classmethod
    def normalize_header_rows(cls, value: List[List[str]]) -> List[List[str]]:
        return [[str(cell or "").strip() for cell in row] for row in value]

    @model_validator(mode="after")
    def validate_shape(self) -> "TableData":
        column_count = len(self.columns)
        if self.header:
            if column_count and len(self.header) != column_count:
                raise ValueError("table header length must match declared columns")
            if not column_count:
                self.columns = [
                    TableColumn(index=index, name=name)
                    for index, name in enumerate(self.header)
                ]
                column_count = len(self.columns)

        if self.header_rows:
            if not self.header:
                self.header = list(self.header_rows[-1])
            for header_row in self.header_rows:
                if column_count and len(header_row) > column_count:
                    raise ValueError("table header row has more cells than declared columns")

        if column_count:
            for row in self.rows:
                if len(row) > column_count:
                    raise ValueError("table row has more cells than declared columns")
        elif self.rows:
            raise ValueError("table rows require a header or declared columns")

        if self.cells and len(self.cells) != len(self.header_rows) + len(self.rows):
            raise ValueError("table cells must align with header_rows and rows")
        return self

    @classmethod
    def from_matrix(
        cls,
        matrix: Sequence[Sequence[object]],
        *,
        caption: Optional[str] = None,
        table_id: Optional[str] = None,
        header_row_count: int = 1,
        cells: Optional[List[List[TableCell]]] = None,
    ) -> "TableData":
        """Build a table from extractor rows, using the first row as header."""
        normalized = [
            [str(cell or "").replace("\n", " ").strip() for cell in row]
            for row in matrix
        ]
        if header_row_count < 0:
            raise ValueError("header_row_count must be non-negative")
        if not normalized:
            return cls(table_id=table_id or str(uuid.uuid4()), caption=caption)

        width = max(len(row) for row in normalized)
        header_row_count = min(header_row_count, len(normalized))
        header_rows = normalized[:header_row_count]
        header = (
            header_rows[-1] + [""] * (width - len(header_rows[-1]))
            if header_rows
            else []
        )
        if cells is None:
            cells = [
                [
                    TableCell(row_index=row_index, column_index=column_index, text=value)
                    for column_index, value in enumerate(row)
                ]
                for row_index, row in enumerate(normalized)
            ]
        return cls(
            table_id=table_id or str(uuid.uuid4()),
            caption=caption,
            columns=[
                TableColumn(index=index, name=header[index] if index < len(header) else "")
                for index in range(width)
            ],
            header=header,
            header_rows=header_rows,
            rows=normalized[header_row_count:],
            cells=cells,
        )

    def to_markdown(self) -> str:
        """Compatibility renderer; loaders do not use Markdown as table storage."""
        if not self.columns:
            return ""
        header = self.header or [column.name for column in self.columns]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(self.columns)) + " |",
        ]
        for row in self.rows:
            padded = row + [""] * (len(self.columns) - len(row))
            lines.append("| " + " | ".join(padded) + " |")
        return "\n".join(lines)


class ParsedBlock(BaseModel):
    """A parser-level block with source location and hierarchy context."""

    block_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    block_index: int = Field(ge=0)
    block_type: BlockType
    text: Optional[str] = None
    section_path: List[str] = Field(default_factory=list)
    section_id: Optional[str] = None
    heading_level: Optional[int] = Field(default=None, ge=1, le=6)
    region: DocumentRegion = DocumentRegion.UNKNOWN
    page_number: Optional[int] = Field(default=None, ge=1)
    bbox: Optional[BoundingBox] = None
    table: Optional[TableData] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        value = value.strip() if value else None
        return value or None

    @field_validator("section_path")
    @classmethod
    def normalize_section_path(cls, value: List[str]) -> List[str]:
        return [part.strip() for part in value if part and part.strip()]

    @model_validator(mode="after")
    def validate_block_shape(self) -> "ParsedBlock":
        if self.block_type == BlockType.TABLE and self.table is None:
            raise ValueError("table blocks require TableData")
        if self.block_type != BlockType.TABLE and self.table is not None:
            raise ValueError("only table blocks may contain TableData")
        if self.block_type == BlockType.HEADING:
            if not self.text:
                raise ValueError("heading blocks require text")
            if self.heading_level is None:
                raise ValueError("heading blocks require heading_level")
        elif self.heading_level is not None:
            raise ValueError("heading_level is only valid for heading blocks")
        return self

    @classmethod
    def section_identifier(cls, source: str, section_path: Sequence[str]) -> Optional[str]:
        """Stable, source-scoped ID for a known hierarchy path."""
        if not section_path:
            return None
        raw = f"{source}\x1f{'\x1f'.join(section_path)}".encode("utf-8")
        return f"section_{sha1(raw).hexdigest()[:16]}"


__all__ = [
    "BlockType",
    "BoundingBox",
    "DocumentRegion",
    "ParsedBlock",
    "TableColumn",
    "TableCell",
    "TableData",
]
