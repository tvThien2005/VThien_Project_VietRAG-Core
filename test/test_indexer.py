import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.http import models

from src.config import get_config
from src.ingestion.indexer import QdrantIndexer
from src.schemas.document import Chunk


class FakeEmbedder:
    model_name = "test-embedding-model"
    dimension = 2

    @staticmethod
    def embed_chunks(chunks, batch_size=None, show_progress_bar=True):
        return [[float(index + 1), 0.5] for index, _ in enumerate(chunks)]


class FakeQdrantClient:
    def __init__(self):
        self.collections = {}
        self.indexes = []
        self.created = []

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.collections]
        )

    def create_collection(self, collection_name, vectors_config, sparse_vectors_config):
        self.created.append(collection_name)
        self.collections[collection_name] = {
            "vectors": vectors_config,
            "sparse": sparse_vectors_config,
            "points": [],
        }
        return True

    def create_payload_index(self, **kwargs):
        self.indexes.append(kwargs)
        return True

    def upsert(self, collection_name, points, wait=True):
        existing = {str(point.id): point for point in self.collections[collection_name]["points"]}
        existing.update({str(point.id): point for point in points})
        self.collections[collection_name]["points"] = list(existing.values())

    def delete(self, collection_name, points_selector, wait=True):
        removed = {str(point_id) for point_id in points_selector.points}
        self.collections[collection_name]["points"] = [
            point for point in self.collections[collection_name]["points"]
            if str(point.id) not in removed
        ]

    def get_collection(self, collection_name):
        collection = self.collections[collection_name]
        return SimpleNamespace(
            points_count=len(collection["points"]),
            status="green",
            indexed_vectors_count=len(collection["points"]),
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=collection["vectors"],
                    sparse_vectors=collection["sparse"],
                )
            ),
        )

    def scroll(self, collection_name, offset=None, limit=256, **kwargs):
        points = self.collections[collection_name]["points"]
        scroll_filter = kwargs.get("scroll_filter")
        if scroll_filter and scroll_filter.must:
            condition = scroll_filter.must[0]
            value = condition.match.value
            points = [point for point in points if (point.payload or {}).get(condition.key) == value]
        start = offset or 0
        page = points[start : start + limit]
        next_offset = start + len(page) if start + len(page) < len(points) else None
        return page, next_offset

    def query_points(self, collection_name, **kwargs):
        points = self.collections[collection_name]["points"]
        return SimpleNamespace(points=points[:1])


def make_chunk(text, doc_id="doc-a"):
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        text=text,
        chunk_index=0,
        metadata={
            "source": f"{doc_id}.pdf",
            "file_name": f"{doc_id}.pdf",
            "section_id": "section-a",
            "section_path": ["Chapter 1", "Section A"],
            "chunking_version": "structured-v1",
            "tokenizer": "BAAI/bge-m3",
            "token_limit": 512,
            "chunk_index_in_section": 0,
            "block_ids": ["source-block"],
            "source_block_ids": ["source-block"],
            "block_types": ["paragraph"],
            "chunk_has_table": False,
        },
    )


class TestVersionedQdrantIndex(unittest.TestCase):
    def setUp(self):
        self.indexer = QdrantIndexer.__new__(QdrantIndexer)
        self.indexer.cfg = get_config(reload=True)
        self.indexer.host = "localhost"
        self.indexer.port = 6333
        self.indexer.client = FakeQdrantClient()
        self.indexer.bm25_vector_name = "bm25"
        self.indexer.bm25_model = "Qdrant/bm25"

    def test_creates_new_named_dense_and_sparse_index_and_verifies_payload(self):
        chunks = [make_chunk("first document content"), make_chunk("second document content", "doc-b")]
        original_collections = {"vietnamese_docs": {"points": []}}
        self.indexer.client.collections.update(original_collections)

        count = self.indexer.index_chunks_versioned(
            chunks,
            embedder=FakeEmbedder(),
            batch_size=1,
            collection_name="isolated-v1-test",
        )

        self.assertEqual(count, 2)
        self.assertIn("vietnamese_docs", self.indexer.client.collections)
        self.assertEqual(self.indexer.client.created, ["isolated-v1-test"])
        collection = self.indexer.client.collections["isolated-v1-test"]
        self.assertEqual(set(collection["vectors"]), {"dense"})
        self.assertEqual(set(collection["sparse"]), {"bm25"})
        points = collection["points"]
        self.assertEqual({point.id for point in points}, {chunk.chunk_id for chunk in chunks})
        for point, chunk in zip(points, chunks):
            self.assertEqual(set(point.vector), {"dense", "bm25"})
            self.assertIsInstance(point.vector["bm25"], models.Document)
            self.assertEqual(point.payload["doc_id"], chunk.doc_id)
            self.assertEqual(point.payload["chunking_version"], "structured-v1")
            self.assertEqual(point.payload["embedding_model"], FakeEmbedder.model_name)
        self.assertIn("table_id", {item["field_name"] for item in self.indexer.client.indexes})

    def test_refuses_to_overwrite_existing_collection(self):
        self.indexer.client.collections["existing"] = {
            "vectors": {},
            "sparse": {},
            "points": [],
        }

        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.indexer.index_chunks_versioned(
                [make_chunk("content")],
                embedder=FakeEmbedder(),
                collection_name="existing",
            )

        self.assertEqual(self.indexer.client.created, [])

    def test_refuses_to_mix_chunk_versions(self):
        first = make_chunk("first")
        second = make_chunk("second", "doc-b")
        second.metadata["chunking_version"] = "structured-v2"

        with self.assertRaisesRegex(ValueError, "one explicit chunking_version"):
            self.indexer.index_chunks_versioned(
                [first, second],
                embedder=FakeEmbedder(),
                collection_name="must-not-exist",
            )

        self.assertEqual(self.indexer.client.created, [])

    def test_appends_to_existing_versioned_collection_without_changing_other_documents(self):
        existing = make_chunk("old document", "doc-old")
        self.indexer.index_chunks_versioned(
            [existing],
            embedder=FakeEmbedder(),
            collection_name="active-versioned",
        )
        incoming = make_chunk("new document", "doc-new")

        count = self.indexer.append_chunks_versioned(
            [incoming],
            embedder=FakeEmbedder(),
            collection_name="active-versioned",
        )

        self.assertEqual(count, 1)
        points = self.indexer.client.collections["active-versioned"]["points"]
        self.assertEqual({point.payload["doc_id"] for point in points}, {"doc-old", "doc-new"})

    def test_rejects_duplicate_document_by_default(self):
        existing = make_chunk("original", "doc-duplicate")
        self.indexer.index_chunks_versioned(
            [existing],
            embedder=FakeEmbedder(),
            collection_name="active-versioned",
        )

        with self.assertRaises(FileExistsError):
            self.indexer.append_chunks_versioned(
                [make_chunk("replacement", "doc-duplicate")],
                embedder=FakeEmbedder(),
                collection_name="active-versioned",
            )


if __name__ == "__main__":
    unittest.main()
