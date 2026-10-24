import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import api_client
from api.routes import ChatRequest, chat, system_status


class TestSystemStatus(TestCase):
    @patch("api.routes.QdrantClient")
    @patch("api.routes.get_config")
    def test_status_uses_exact_count_and_indexed_embedding_model(self, get_config, qdrant_client):
        config = SimpleNamespace(
            vector_store=SimpleNamespace(
                host="localhost",
                port=6333,
                api_key=None,
                timeout=30,
                collection_name="demo_collection",
            ),
            embedding=SimpleNamespace(model_name="configured-model"),
            app=SimpleNamespace(version="0.2.0"),
        )
        get_config.return_value = config
        client = qdrant_client.return_value
        client.count.return_value.count = 724
        client.scroll.return_value = ([SimpleNamespace(
            payload={"embedding_model": "indexed-model"},
        )], None)

        result = system_status()

        client.count.assert_called_once_with(
            collection_name="demo_collection",
            exact=True,
        )
        self.assertEqual(result["total_vectors"], 724)
        self.assertEqual(result["embedding_model"], "indexed-model")
        self.assertEqual(result["collection"], "demo_collection")
        client.close.assert_called_once()


class TestChatConfig(TestCase):
    def test_retrieval_options_default_to_vector_without_reranking(self):
        self.assertEqual(api_client.DEFAULT_CONFIG["search_mode"], "vector")
        self.assertFalse(api_client.DEFAULT_CONFIG["use_hybrid"])
        self.assertFalse(api_client.DEFAULT_CONFIG["rerank"])

    @patch("app.api_client._request")
    @patch("app.api_client.MOCK_MODE", False)
    def test_ask_sends_selected_hybrid_and_rerank_options(self, request):
        request.return_value = {"answer": "ok"}
        config = {
            **api_client.DEFAULT_CONFIG,
            "search_mode": "hybrid",
            "use_hybrid": True,
            "rerank": True,
        }

        api_client.ask("token", "question", None, config)

        payload = request.call_args.kwargs["json"]
        self.assertEqual(payload["search_mode"], "hybrid")
        self.assertTrue(payload["rerank"])
        self.assertFalse(payload["hyde"])


class TestChatRoute(TestCase):

    def test_chat_uses_yaml_values_when_ui_does_not_override_them(self):
        rag_response = SimpleNamespace(answer="ok", sources=[], latency_seconds=0.01)
        pipeline = Mock()
        pipeline.query.return_value = rag_response
        config = SimpleNamespace(
            retrieval=SimpleNamespace(
                top_k=10,
                hybrid_search=SimpleNamespace(enabled=False),
                reranker=SimpleNamespace(enabled=False, top_n=4),
            )
        )

        with patch("api.routes.get_config", return_value=config), \
             patch("api.routes.get_rag_pipeline", return_value=pipeline):
            chat(ChatRequest(question="Câu hỏi demo"))

        pipeline.query.assert_called_once_with(
            query_text="Câu hỏi demo",
            top_k=10,
            top_n=4,
            use_rerank=False,
            use_hybrid=False,
        )

    def test_chat_calls_existing_pipeline_and_returns_sidebar_sources(self):
        chunk = SimpleNamespace(
            chunk_id="chunk-1",
            text="Nội dung liên quan.",
            score=0.87,
            page_number=4,
            metadata={"file_name": "paper.pdf"},
        )
        rag_response = SimpleNamespace(
            answer="Câu trả lời [1].",
            sources=[chunk],
            latency_seconds=0.42,
        )
        pipeline = Mock()
        pipeline.query.return_value = rag_response

        config = SimpleNamespace(
            retrieval=SimpleNamespace(
                top_k=10,
                hybrid_search=SimpleNamespace(enabled=False),
                reranker=SimpleNamespace(enabled=False, top_n=4),
            )
        )
        with patch("api.routes.get_config", return_value=config), \
             patch("api.routes.get_rag_pipeline", return_value=pipeline):
            result = chat(ChatRequest(
                question="Câu hỏi demo",
                search_mode="hybrid",
                top_k=3,
                rerank=True,
            ))

        pipeline.query.assert_called_once_with(
            query_text="Câu hỏi demo",
            top_k=3,
            top_n=4,
            use_rerank=True,
            use_hybrid=True,
        )
        self.assertEqual(result["answer"], "Câu trả lời [1].")
        self.assertEqual(result["sources"][0]["doc_name"], "paper.pdf")
        self.assertEqual(result["sources"][0]["page"], 4)
        self.assertEqual(result["sources"][0]["score"], 0.87)
        self.assertEqual(result["metrics"]["total_ms"], 420)

    def test_chat_rejects_bm25_until_pipeline_supports_it(self):
        with self.assertRaises(HTTPException) as error:
            chat(ChatRequest(question="Câu hỏi", search_mode="bm25"))

        self.assertEqual(error.exception.status_code, 422)
