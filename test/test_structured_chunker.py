import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunkers.section_builder import SectionBuilder
from src.ingestion.chunkers.structured_chunker import StructuredChunker
from src.schemas.blocks import BlockType, ParsedBlock, TableData


class FakeTokenizer:
    name_or_path = "test-word-tokenizer"

    @staticmethod
    def _offsets(text):
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]

    def encode(self, text, add_special_tokens=False):
        return list(range(len(self._offsets(text))))

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"offset_mapping": self._offsets(text)}


class CharacterTokenizer(FakeTokenizer):
    @staticmethod
    def _offsets(text):
        return [(index, index + 1) for index in range(len(text))]


def block(
    index,
    block_type,
    text=None,
    *,
    level=None,
    page=None,
    table=None,
    source="sample.pdf",
    metadata=None,
):
    return ParsedBlock(
        block_index=index,
        block_type=block_type,
        text=text,
        heading_level=level,
        page_number=page,
        table=table,
        metadata={
            "source": source,
            "file_name": Path(source).name,
            "file_type": Path(source).suffix.lstrip("."),
            **(metadata or {}),
        },
    )


class TestSectionBuilder(unittest.TestCase):
    def test_hierarchy_persists_across_pages_without_inventing_chapter(self):
        blocks = [
            block(0, BlockType.HEADING, "Overview", level=1, page=1, metadata={"chapter": "Trang 1"}),
            block(1, BlockType.PARAGRAPH, "Content on page two.", page=2, metadata={"chapter": "Trang 2"}),
            block(2, BlockType.HEADING, "2.1 Scope", level=2, page=2),
            block(3, BlockType.TABLE, table=TableData.from_matrix([["A", "B"]], header_row_count=0), page=3),
            block(4, BlockType.HEADING, "Chapter 3: Results", level=1, page=4),
            block(5, BlockType.PARAGRAPH, "Result content.", page=5),
        ]

        result = SectionBuilder().build(blocks)
        by_index = {item.block_index: item for item in result.blocks}

        self.assertEqual(by_index[1].section_path, ["Overview"])
        self.assertEqual(by_index[1].metadata["chapter"], "")
        self.assertEqual(by_index[1].metadata["section_title"], "Overview")
        self.assertEqual(by_index[3].section_path, ["Overview", "2.1 Scope"])
        self.assertEqual(by_index[3].section_id, by_index[2].section_id)
        self.assertEqual(
            by_index[3].metadata["parent_section_id"],
            by_index[2].metadata["parent_section_id"],
        )
        self.assertEqual(by_index[5].section_path, ["Chapter 3: Results"])
        self.assertEqual(by_index[5].metadata["chapter"], "Chapter 3: Results")
        self.assertEqual(by_index[5].section_id, by_index[4].section_id)
        self.assertNotIn("Trang 5", by_index[5].section_path)
        self.assertEqual(len(result.sections), 3)

    def test_numeric_heading_is_not_promoted_to_chapter(self):
        result = SectionBuilder().build(
            [block(0, BlockType.HEADING, "29", level=1, page=40)]
        )

        self.assertEqual(result.blocks[0].metadata["chapter"], "")
        self.assertEqual(result.blocks[0].section_path, ["29"])


class TestStructuredChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = StructuredChunker(
            max_tokens=12,
            overlap_tokens=2,
            tokenizer=FakeTokenizer(),
        )

    def test_chunks_never_combine_different_sections_and_heading_joins_first_content(self):
        blocks = [
            block(0, BlockType.HEADING, "Chapter 1: Intro", level=1, page=1),
            block(1, BlockType.PARAGRAPH, "One short paragraph.", page=1),
            block(2, BlockType.HEADING, "Chapter 2: Results", level=1, page=2),
            block(3, BlockType.PARAGRAPH, "A second section paragraph.", page=2),
        ]

        chunks = self.chunker.split_blocks(blocks, doc_id="doc-id")

        self.assertEqual(len(chunks), 2)
        self.assertIn("# Chapter 1: Intro", chunks[0].text)
        self.assertIn("One short paragraph.", chunks[0].text)
        self.assertIn("# Chapter 2: Results", chunks[1].text)
        self.assertTrue(all(chunk.metadata["doc_id"] == "doc-id" for chunk in chunks))
        self.assertTrue(all(len(chunk.metadata["section_path"]) == 1 for chunk in chunks))
        self.assertNotEqual(chunks[0].metadata["section_id"], chunks[1].metadata["section_id"])
        self.assertTrue(all(not chunk.metadata.get("heading_only", False) for chunk in chunks))
        self.assertTrue(all(chunk.metadata["chunk_has_table"] is False for chunk in chunks))

    def test_long_paragraph_is_split_at_sentences_then_token_boundaries_with_declared_overlap(self):
        paragraph = (
            "First sentence has four words. "
            "Second sentence has four words. "
            "Third sentence has many words that exceed every individual chunk size and must be "
            "split precisely at tokenizer offsets without dropping any original words."
        )
        blocks = [block(0, BlockType.PARAGRAPH, paragraph, page=1)]

        chunks = self.chunker.split_blocks(blocks, doc_id="doc-long")

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["token_count"] <= 12 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["token_limit_exceeded"] is False for chunk in chunks))
        self.assertTrue(all(chunk.metadata["chunk_has_table"] is False for chunk in chunks))
        self.assertTrue(all(chunk.metadata["page_start"] == 1 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["page_end"] == 1 for chunk in chunks))
        non_overlapping = StructuredChunker(
            max_tokens=12,
            overlap_tokens=0,
            tokenizer=FakeTokenizer(),
        ).split_blocks(blocks, doc_id="doc-no-overlap")
        reconstructed = "".join(
            fragment["content"]
            for chunk in non_overlapping
            for fragment in chunk.metadata["source_fragments"]
        )
        self.assertEqual(reconstructed, paragraph)

    def test_overlap_is_token_bounded_and_marked_when_recursive_separators_cannot_split(self):
        chunker = StructuredChunker(
            max_tokens=12,
            overlap_tokens=2,
            tokenizer=CharacterTokenizer(),
        )
        text = "abcdefghijklmnopqrstuvwx"

        chunks = chunker.split_blocks([block(0, BlockType.PARAGRAPH, text)])

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["token_count"] <= 12 for chunk in chunks))
        self.assertTrue(any(chunk.metadata.get("overlap_applied") for chunk in chunks))
        spans = [
            fragment
            for chunk in chunks
            for fragment in chunk.metadata["source_fragments"]
        ]
        self.assertEqual(spans[0]["start_char"], 0)
        self.assertEqual(spans[-1]["end_char"], len(text))
        self.assertTrue(
            any(
                current["start_char"] < previous["end_char"]
                for previous, current in zip(spans, spans[1:])
            )
        )

    def test_long_section_paragraph_does_not_infer_chapter_from_sentence_or_page(self):
        paragraph = "ordinary prose " * 35
        blocks = [
            block(0, BlockType.PARAGRAPH, paragraph, page=7),
            block(
                1,
                BlockType.PARAGRAPH,
                "Chapter 9 explains the result.",
                page=7,
                metadata={"section_title": "Chapter 9", "chapter": "Trang 7"},
            ),
        ]

        chunks = self.chunker.split_blocks(blocks)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.metadata["chapter"] == "" for chunk in chunks))
        self.assertTrue(all("Trang 7" not in chunk.metadata["section_path"] for chunk in chunks))
        self.assertTrue(
            all("Chapter 9 explains the result." not in chunk.metadata["section_path"] for chunk in chunks)
        )

    def test_tables_are_atomic_and_only_table_chunks_claim_table_content(self):
        long_table = TableData.from_matrix(
            [[f"Cell {row} value"] for row in range(20)],
            caption="Results",
            header_row_count=0,
        )
        blocks = [
            block(0, BlockType.PARAGRAPH, "Intro text."),
            block(1, BlockType.TABLE, table=long_table),
            block(2, BlockType.PARAGRAPH, "Closing text."),
        ]

        chunks = self.chunker.split_blocks(blocks)
        table_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_has_table"]]
        prose_chunks = [chunk for chunk in chunks if not chunk.metadata["chunk_has_table"]]

        self.assertEqual(len(table_chunks), 1)
        self.assertGreater(table_chunks[0].metadata["token_count"], 12)
        self.assertTrue(table_chunks[0].metadata["token_limit_exceeded"])
        self.assertEqual(
            table_chunks[0].metadata["oversize_reason"],
            "table_is_an_indivisible_structural_unit",
        )
        self.assertEqual(table_chunks[0].metadata["source_table"]["caption"], "Results")
        self.assertEqual(table_chunks[0].metadata["table_caption"], "Results")
        self.assertEqual(table_chunks[0].metadata["table_caption_status"], "present")
        self.assertTrue(all(chunk.metadata["chunk_has_table"] is False for chunk in prose_chunks))
        self.assertIn("Intro text.", prose_chunks[0].text)
        self.assertIn("Closing text.", prose_chunks[-1].text)

        uncaptioned = self.chunker.split_blocks(
            [block(0, BlockType.TABLE, table=TableData.from_matrix([["A"]]))]
        )[0]
        self.assertIsNone(uncaptioned.metadata["table_caption"])
        self.assertEqual(uncaptioned.metadata["table_caption_status"], "missing")

    def test_structural_metadata_is_complete_and_source_traceable(self):
        blocks = [
            block(0, BlockType.HEADING, "Chapter 4: Data", level=1, page=8),
            block(1, BlockType.PARAGRAPH, "Data paragraph.", page=9),
        ]

        chunk = self.chunker.split_blocks(blocks)[0]

        self.assertEqual(chunk.metadata["source"], "sample.pdf")
        self.assertEqual(chunk.metadata["file_name"], "sample.pdf")
        self.assertEqual(chunk.metadata["doc_id"], chunk.doc_id)
        self.assertEqual(chunk.metadata["chunk_id"], chunk.chunk_id)
        self.assertEqual(chunk.metadata["section_path"], ["Chapter 4: Data"])
        self.assertEqual(chunk.metadata["chapter"], "Chapter 4: Data")
        self.assertEqual(chunk.metadata["page_start"], 8)
        self.assertEqual(chunk.metadata["page_end"], 9)
        self.assertEqual(chunk.metadata["block_types"], ["heading", "paragraph"])
        self.assertEqual(chunk.metadata["chunking_version"], "structured-v1")
        self.assertTrue(chunk.metadata["source_block_ids"])
        self.assertTrue(chunk.metadata["source_content"])


class TestIngestionPipelineStructuredRouting(unittest.TestCase):
    def test_run_routes_parsed_pdf_blocks_to_structured_chunker(self):
        from src.ingestion.pipeline import IngestionPipeline

        parsed = [block(0, BlockType.PARAGRAPH, "Structured source.")]
        fake_chunk = Mock()
        structured_chunker = Mock()
        structured_chunker.split_blocks.return_value = [fake_chunk]
        loader = Mock()
        loader.load_blocks.return_value = parsed

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "input.pdf"
            path.touch()
            pipeline = IngestionPipeline.__new__(IngestionPipeline)
            pipeline.cfg = Mock()
            pipeline.cfg.data.supported_extensions = [".pdf"]
            pipeline.cfg.data.processed_path = Path(temp_dir)
            pipeline.chunker = Mock()
            pipeline.structured_chunker = structured_chunker
            with patch("src.ingestion.pipeline.get_loader", return_value=loader):
                result = pipeline.run(path, save_to_processed=False)

        self.assertEqual(result, [fake_chunk])
        loader.load_blocks.assert_called_once_with(path)
        structured_chunker.split_blocks.assert_called_once_with(parsed)
        pipeline.chunker.split_documents.assert_not_called()


if __name__ == "__main__":
    unittest.main()
