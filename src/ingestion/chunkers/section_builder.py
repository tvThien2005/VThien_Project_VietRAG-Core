"""Build a stable hierarchy from normalized parser blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.schemas.blocks import BlockType, ParsedBlock

_EXPLICIT_CHAPTER = re.compile(r"^(?:chương|chapter)\s+\d+\b", re.IGNORECASE)


@dataclass
class BuiltSection:
    section_id: Optional[str]
    parent_section_id: Optional[str]
    title: Optional[str]
    level: int
    section_path: List[str]
    chapter: str
    blocks: List[ParsedBlock] = field(default_factory=list)


@dataclass
class SectionBuildResult:
    blocks: List[ParsedBlock]
    sections: List[BuiltSection]


class SectionBuilder:
    """Apply heading hierarchy to blocks without deriving titles from prose/pages."""

    def build(self, blocks: Sequence[ParsedBlock]) -> SectionBuildResult:
        enriched_blocks: List[ParsedBlock] = []
        sections: List[BuiltSection] = []
        heading_stack: List[Tuple[int, BuiltSection]] = []
        root_section: Optional[BuiltSection] = None
        current_source: Optional[str] = None

        for block in blocks:
            source = str(block.metadata.get("source", ""))
            if current_source is not None and source != current_source:
                heading_stack = []
                root_section = None
            current_source = source

            if block.block_type == BlockType.HEADING:
                level = block.heading_level
                if level is None:
                    raise ValueError(
                        f"Heading block {block.block_id} is missing heading_level."
                    )
                heading_stack = [
                    (parent_level, section)
                    for parent_level, section in heading_stack
                    if parent_level < level
                ]
                parent = heading_stack[-1][1] if heading_stack else None
                parent_path = parent.section_path if parent else []
                section_path = [*parent_path, block.text or ""]
                section = BuiltSection(
                    section_id=ParsedBlock.section_identifier(source, section_path),
                    parent_section_id=parent.section_id if parent else None,
                    title=block.text,
                    level=level,
                    section_path=section_path,
                    chapter=(
                        section_path[0]
                        if section_path and _EXPLICIT_CHAPTER.match(section_path[0])
                        else ""
                    ),
                )
                sections.append(section)
                heading_stack.append((level, section))
            else:
                section = heading_stack[-1][1] if heading_stack else None
                if section is None:
                    if root_section is None:
                        root_section = BuiltSection(
                            section_id=None,
                            parent_section_id=None,
                            title=None,
                            level=0,
                            section_path=[],
                            chapter="",
                        )
                        sections.append(root_section)
                    section = root_section

            section_path = list(section.section_path)
            metadata: Dict[str, Any] = dict(block.metadata)
            metadata.update(
                {
                    "section_path": section_path,
                    "section_id": section.section_id,
                    "parent_section_id": section.parent_section_id,
                    "section_title": section.title or "",
                    "chapter": section.chapter,
                    "breadcrumb": " > ".join(section_path),
                }
            )
            enriched = block.model_copy(
                update={
                    "section_path": section_path,
                    "section_id": section.section_id,
                    "metadata": metadata,
                }
            )
            section.blocks.append(enriched)
            enriched_blocks.append(enriched)

        return SectionBuildResult(blocks=enriched_blocks, sections=sections)


__all__ = ["BuiltSection", "SectionBuildResult", "SectionBuilder"]
