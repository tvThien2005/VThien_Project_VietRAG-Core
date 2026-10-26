import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ContextExpansionConfig
from src.retrieval.context_expander import ContextExpander, create_context_tokenizer
from src.schemas.blocks import TableData
from src.schemas.response import RetrievedChunk


class WordTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        return text.split()


class FilteredScrollClient:
    def __init__(self, points):
        self.points = points
        self.last_filter = None

    def scroll(self, **kwargs):
        self.last_filter = kwargs["scroll_filter"]
        matched = []
        for point in self.points:
            payload = point.payload
            include = True
            for condition in self.last_filter.must:
                value = payload.get(condition.key)
                if condition.match is not None:
                    include = include and value == condition.match.value
                if condition.range is not None:
                    if value is None:
                        include = False
                    else:
                        if condition.range.gte is not None:
                            include = include and value >= condition.range.gte
                        if condition.range.lte is not None:
                            include = include and value <= condition.range.lte
            if include:
                matched.append(point)
        return matched[: kwargs["limit"]], None


def stored_point(chunk_id, text, doc_id, section_id, index, page=1):
    return SimpleNamespace(
        id=chunk_id,
        payload={
            "text": text,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source": f"{doc_id}.pdf",
            "file_name": f"{doc_id}.pdf",
            "section_id": section_id,
            "section_path": ["Chapter 1", "Section A"],
            "section_title": "Section A",
            "chunk_index_in_section": index,
            "chunk_index": index,
            "page_start": page,
            "page_end": page,
        },
    )


class TestContextExpansion(unittest.TestCase):
    def test_default_groq_context_uses_configured_model_tokenizer(self):
        tokenizer = create_context_tokenizer()

        self.assertEqual(tokenizer.name, "o200k_harmony")

    def test_neighbors_stay_in_the_seed_document_and_section(self):
        points = [
            stored_point("previous", "before seed", "doc-a", "section-a", 0),
            stored_point("seed", "seed content", "doc-a", "section-a", 1),
            stored_point("next", "after seed", "doc-a", "section-a", 2),
            stored_point("other-section", "unrelated section", "doc-a", "section-b", 1),
            stored_point("other-document", "unrelated document", "doc-b", "section-a", 2),
        ]
        client = FilteredScrollClient(points)
        seed = RetrievedChunk(
            chunk_id="seed",
            doc_id="doc-a",
            text="seed content",
            score=0.9,
            rank=1,
            metadata=points[1].payload,
        )
        expander = ContextExpander(
            client=client,
            collection_name="versioned-test",
            tokenizer=WordTokenizer(),
            config=ContextExpansionConfig(
                enabled=True,
                neighbor_window=1,
                max_prompt_tokens=10_000,
            ),
        )

        expanded = expander.expand("question", [seed])

        self.assertEqual([chunk.chunk_id for chunk in expanded], ["previous", "seed", "next"])
        self.assertTrue(all(chunk.metadata["section_id"] == "section-a" for chunk in expanded))
        self.assertTrue(all(chunk.metadata["source"] == "doc-a.pdf" for chunk in expanded))
        self.assertEqual(expanded[1].rank, 1)
        self.assertTrue(expanded[0].metadata["is_context_expansion"])
        self.assertEqual(client.last_filter.must[0].key, "doc_id")
        self.assertEqual(client.last_filter.must[1].key, "section_id")

    def test_neighbors_keep_the_authenticated_owner_scope(self):
        points = [
            stored_point("previous", "before seed", "doc-a", "section-a", 0),
            stored_point("seed", "seed content", "doc-a", "section-a", 1),
            stored_point("other-owner", "private content", "doc-a", "section-a", 2),
        ]
        points[0].payload["owner_user_id"] = "user-a"
        points[1].payload["owner_user_id"] = "user-a"
        points[2].payload["owner_user_id"] = "user-b"
        client = FilteredScrollClient(points)
        seed = RetrievedChunk(
            chunk_id="seed", doc_id="doc-a", text="seed content", score=0.9, rank=1,
            metadata=points[1].payload,
        )
        expander = ContextExpander(
            client=client,
            collection_name="versioned-test",
            tokenizer=WordTokenizer(),
            config=ContextExpansionConfig(enabled=True, neighbor_window=1, max_prompt_tokens=10_000),
        )

        expanded = expander.expand("question", [seed], owner_user_id="user-a")

        self.assertEqual([chunk.chunk_id for chunk in expanded], ["previous", "seed"])
        self.assertEqual(client.last_filter.must[0].key, "owner_user_id")
        self.assertEqual(client.last_filter.must[0].match.value, "user-a")

    def test_full_prompt_budget_limits_context_growth(self):
        shared = "word " * 80
        points = [
            stored_point("previous", shared, "doc-a", "section-a", 0),
            stored_point("seed", shared, "doc-a", "section-a", 1),
            stored_point("next", shared, "doc-a", "section-a", 2),
        ]
        seed = RetrievedChunk(
            chunk_id="seed",
            doc_id="doc-a",
            text=shared,
            score=0.9,
            rank=1,
            metadata=points[1].payload,
        )
        expander = ContextExpander(
            client=FilteredScrollClient(points),
            collection_name="versioned-test",
            tokenizer=WordTokenizer(),
            config=ContextExpansionConfig(
                enabled=True,
                neighbor_window=1,
                max_prompt_tokens=520,
            ),
        )

        expanded = expander.expand("question", [seed])

        self.assertEqual({chunk.chunk_id for chunk in expanded}, {"seed", "previous"})
        self.assertTrue(expander._prompt_fits("question", expanded))
        self.assertFalse(
            expander._prompt_fits(
                "question",
                [
                    *expanded,
                    next(
                        chunk
                        for chunk in expander._neighbors(seed)
                        if chunk.chunk_id == "next"
                    ),
                ],
            )
        )

    def test_large_table_keeps_caption_and_header_while_selecting_relevant_rows(self):
        rows = [
            ["target row", "relevant result " + "data " * 20],
            *[
                [f"unrelated row {index}", "other values " + "data " * 20]
                for index in range(1, 9)
            ],
        ]
        table = TableData.from_matrix(
            [["Metric", "Value"], *rows],
            caption="Table 4: Model results",
        )
        full_table_text = ContextExpander._table_text(
            table,
            list(range(len(table.rows))),
        )
        payload = {
            **stored_point("table-chunk", full_table_text, "doc-a", "section-a", 0).payload,
            "chunk_has_table": True,
            "table_id": table.table_id,
            "table_caption": table.caption,
            "table_caption_status": "present",
            "source_table": table.model_dump(mode="json"),
        }
        seed = RetrievedChunk(
            chunk_id="table-chunk",
            doc_id="doc-a",
            text=full_table_text,
            score=0.9,
            rank=1,
            metadata=payload,
        )
        expander = ContextExpander(
            client=FilteredScrollClient([]),
            collection_name="versioned-test",
            tokenizer=WordTokenizer(),
            config=ContextExpansionConfig(
                enabled=True,
                neighbor_window=0,
                max_prompt_tokens=380,
            ),
        )

        expanded = expander.expand("target result", [seed])

        self.assertEqual(len(expanded), 1)
        self.assertIn("Table 4: Model results", expanded[0].text)
        self.assertIn("| Metric | Value |", expanded[0].text)
        self.assertIn("target row", expanded[0].text)
        self.assertEqual(expanded[0].metadata["table_id"], table.table_id)
        self.assertEqual(expanded[0].metadata["table_caption_status"], "present")
        self.assertTrue(expanded[0].metadata["context_table_excerpt"])
        self.assertTrue(expander._prompt_fits("target result", expanded))


if __name__ == "__main__":
    unittest.main()
