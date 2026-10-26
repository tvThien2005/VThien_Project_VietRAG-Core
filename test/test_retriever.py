import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas.response import RetrievedChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.config import get_config
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.qdrant_filters import build_payload_filter


class TestRetrieverSchemas(unittest.TestCase):
    def test_retrieved_chunk_properties(self):
        chunk = RetrievedChunk(
            chunk_id="test-chunk-123",
            text="Đây là đoạn văn bản thử nghiệm.",
            score=0.92,
            rank=1,
            metadata={
                "file_name": "tailieu.pdf",
                "page_number": 5,
                "section_title": "Giới thiệu",
            },
        )
        self.assertEqual(chunk.file_name, "tailieu.pdf")
        self.assertEqual(chunk.page_number, 5)
        self.assertEqual(chunk.section_title, "Giới thiệu")
        self.assertAlmostEqual(chunk.score, 0.92)

    def test_owner_filter_is_a_mandatory_payload_condition(self):
        query_filter = build_payload_filter(owner_user_id="user-a", doc_id="doc-a")

        self.assertEqual([condition.key for condition in query_filter.must], ["owner_user_id", "doc_id"])
        self.assertEqual(query_filter.must[0].match.value, "user-a")


class TestHybridRetriever(unittest.TestCase):
    def test_candidate_settings_are_loaded_from_config(self):
        class FakeVectorRetriever:
            client = object()
            collection_name = "test-collection"

        config = get_config(reload=True).retrieval.hybrid_search
        retriever = HybridRetriever(vector_retriever=FakeVectorRetriever())

        self.assertEqual(retriever.candidate_k, config.candidate_k)
        self.assertEqual(retriever.candidate_multiplier, config.candidate_multiplier)
        self.assertEqual(retriever.min_candidates, config.min_candidates)
        self.assertEqual(retriever.rrf_k, config.rrf_k)

    def test_sparse_ranking_uses_qdrant_bm25_query_results(self):
        class FakeClient:
            def query_points(self, **kwargs):
                self.query = kwargs["query"]
                self.using = kwargs["using"]
                self.query_filter = kwargs.get("query_filter")
                points = [
                    SimpleNamespace(
                        id=chunk_id,
                        score=1.0 / rank,
                        payload={"text": text, "doc_id": "doc-1"},
                    )
                    for rank, (chunk_id, text) in enumerate(
                        [
                            ("relevant", "DenseNet201 đạt độ chính xác 93%."),
                            ("two-terms", "DenseNet201 đạt kết quả."),
                            ("partial", "Mô hình CNN."),
                        ],
                        start=1,
                    )
                ]
                return SimpleNamespace(points=points)

        class FakeVectorRetriever:
            client = FakeClient()
            collection_name = "test-collection"

            def retrieve(self, query, top_k=None):
                return []

        retriever = HybridRetriever(
            vector_retriever=FakeVectorRetriever(),
            dense_weight=0.0,
            sparse_weight=1.0,
        )
        results = retriever.retrieve(
            "DenseNet201 kết quả độ chính xác",
            top_k=3,
            doc_id="doc-1",
        )

        self.assertEqual([chunk.chunk_id for chunk in results], ["relevant", "two-terms", "partial"])
        self.assertEqual(retriever.vector_retriever.client.using, "bm25")
        self.assertEqual(retriever.vector_retriever.client.query.text, "DenseNet201 kết quả độ chính xác")
        self.assertEqual(
            retriever.vector_retriever.client.query_filter.must[0].key,
            "doc_id",
        )

    def test_vector_retriever_uses_named_vector_filter_and_returns_only_points(self):
        class FakeEmbedder:
            model_name = "test-embedding"
            dimension = 2

            @staticmethod
            def embed_query(query):
                return [0.1, 0.2]

        class FakeClient:
            def get_collection(self, collection_name):
                return SimpleNamespace(
                    config=SimpleNamespace(
                        params=SimpleNamespace(
                            vectors={
                                "dense": SimpleNamespace(size=2),
                            }
                        )
                    )
                )

            def scroll(self, **kwargs):
                return [
                    SimpleNamespace(
                        payload={
                            "embedding_model": "test-embedding",
                            "embedding_dimension": 2,
                        }
                    )
                ], None

            def query_points(self, **kwargs):
                self.options = kwargs
                return SimpleNamespace(
                    points=[
                        SimpleNamespace(
                            id="chunk-id",
                            score=0.75,
                            payload={"text": "Retrieved text", "doc_id": "doc-a"},
                        )
                    ]
                )

        fake_client = FakeClient()
        retriever = VectorRetriever(
            embedder=FakeEmbedder(),
            client=fake_client,
            collection_name="versioned-test",
            dense_vector_name="dense",
        )

        results = retriever.retrieve("query", doc_id="doc-a")

        self.assertEqual([chunk.chunk_id for chunk in results], ["chunk-id"])
        self.assertEqual(len(results), 1)
        self.assertEqual(fake_client.options["using"], "dense")
        self.assertEqual(
            fake_client.options["query_filter"].must[0].key,
            "doc_id",
        )

    def test_vector_retriever_rejects_incompatible_embedding_model(self):
        class FakeEmbedder:
            model_name = "query-embedding-model"
            dimension = 2

            @staticmethod
            def embed_query(query):
                return [0.1, 0.2]

        class FakeClient:
            def get_collection(self, collection_name):
                return SimpleNamespace(
                    config=SimpleNamespace(
                        params=SimpleNamespace(
                            vectors={"dense": SimpleNamespace(size=2)}
                        )
                    )
                )

            def scroll(self, **kwargs):
                return [
                    SimpleNamespace(
                        payload={
                            "embedding_model": "different-model",
                            "embedding_dimension": 2,
                        }
                    )
                ], None

            def query_points(self, **kwargs):
                raise AssertionError("Must not query with an incompatible embedding.")

        retriever = VectorRetriever(
            embedder=FakeEmbedder(),
            client=FakeClient(),
            collection_name="incompatible-index",
            dense_vector_name="dense",
        )

        with self.assertRaisesRegex(ValueError, "was indexed with"):
            retriever.retrieve("query")


class TestRetrievalPipelineContextExpansion(unittest.TestCase):
    def test_expansion_starts_from_reranked_results_and_preserves_doc_filter(self):
        candidates = [
            RetrievedChunk(chunk_id="low", text="lower score", score=0.2, rank=1),
            RetrievedChunk(chunk_id="best", text="reranked first", score=0.9, rank=2),
        ]
        reranked = [candidates[1]]
        neighbor = RetrievedChunk(
            chunk_id="neighbor",
            text="same-section neighbor",
            score=0.9,
            rank=2,
        )
        retriever = Mock()
        retriever.retrieve.return_value = candidates
        reranker = Mock()
        reranker.rerank.return_value = reranked
        expander = Mock()
        expander.expand.return_value = [reranked[0], neighbor]
        pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
        pipeline.cfg = SimpleNamespace(
            retrieval=SimpleNamespace(
                top_k=10,
                reranker=SimpleNamespace(top_n=5, enabled=True),
                hybrid_search=SimpleNamespace(enabled=False),
            )
        )
        pipeline.retriever = retriever
        pipeline.hybrid_retriever = Mock()
        pipeline.reranker = reranker
        pipeline.context_expander = expander

        result = pipeline.retrieve(
            "question",
            use_hybrid=False,
            use_rerank=True,
            doc_id="doc-a",
        )

        retriever.retrieve.assert_called_once_with(
            query="question",
            top_k=10,
            doc_id="doc-a",
        )
        reranker.rerank.assert_called_once_with(
            query="question",
            chunks=candidates,
            top_n=5,
        )
        expander.expand.assert_called_once_with("question", reranked)
        self.assertEqual(result, [reranked[0], neighbor])


if __name__ == "__main__":
    unittest.main()
