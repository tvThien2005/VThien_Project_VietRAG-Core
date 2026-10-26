import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import ANY, Mock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import api_client
from api.auth import CurrentUser, UserStore, get_current_user, get_user_store
from api.main import app
from api.routes import ChatRequest, chat, system_status
from src.ingestion.document_registry import DocumentRegistry

TEST_USER = CurrentUser(user_id="user-a", email="a@example.com", full_name="User A")


class TestSystemStatus(TestCase):
    @patch("api.routes.QdrantClient")
    @patch("api.routes.get_config")
    def test_status_uses_approximate_count_and_indexed_embedding_model(self, get_config, qdrant_client):
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

        result = system_status(TEST_USER)

        client.count.assert_called_once_with(
            collection_name="demo_collection",
            count_filter=ANY,
            exact=False,
        )
        self.assertEqual(client.count.call_args.kwargs["count_filter"].must[0].key, "owner_user_id")
        self.assertEqual(client.count.call_args.kwargs["count_filter"].must[0].match.value, "user-a")
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

    @patch("app.api_client._request")
    @patch("app.api_client.DOCUMENTS_API_MODE", True)
    def test_upload_uses_real_documents_api_when_enabled(self, request):
        request.return_value = {"job_id": "job-1", "status": "queued"}

        api_client.upload_document("token", "policy.pdf", b"pdf", "application/pdf")

        self.assertEqual(request.call_args.args[:2], ("POST", "/documents"))
        self.assertEqual(request.call_args.kwargs["data"]["duplicate_policy"], "reject")


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
            chat(ChatRequest(question="Câu hỏi demo"), TEST_USER)

        pipeline.query.assert_called_once_with(
            query_text="Câu hỏi demo",
            top_k=10,
            top_n=4,
            use_rerank=False,
            use_hybrid=False,
            owner_user_id="user-a",
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
            ), TEST_USER)

        pipeline.query.assert_called_once_with(
            query_text="Câu hỏi demo",
            top_k=3,
            top_n=4,
            use_rerank=True,
            use_hybrid=True,
            owner_user_id="user-a",
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


class TestDocumentsRoute(TestCase):
    def test_upload_queues_background_job_and_lists_registry_document(self):
        config = SimpleNamespace(data=SimpleNamespace(supported_extensions=[".pdf", ".docx", ".txt"]))
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = DocumentRegistry(Path(temp_dir) / "registry.sqlite3")
            with patch("api.routes.PROJECT_ROOT", Path(temp_dir)), \
                 patch("api.routes.get_config", return_value=config), \
                 patch("api.routes.get_document_registry", return_value=registry), \
                 patch("api.routes.process_document_job") as worker:
                client = TestClient(app)
                app.dependency_overrides[get_current_user] = lambda: TEST_USER
                try:
                    response = client.post(
                        "/documents",
                        files={"file": ("policy.pdf", b"%PDF-demo", "application/pdf")},
                    )
                    self.assertEqual(response.status_code, 202)
                    document = response.json()
                    self.assertEqual(document["name"], "policy.pdf")
                    self.assertEqual(document["status"], "queued")
                    self.assertTrue(document["job_id"])
                    worker.assert_called_once()
                    self.assertEqual(worker.call_args.kwargs["owner_user_id"], "user-a")

                    listed = client.get("/documents")
                    self.assertEqual(listed.status_code, 200)
                    self.assertEqual(listed.json()[0]["id"], document["id"])
                finally:
                    app.dependency_overrides.clear()

    def test_list_documents_never_returns_another_users_registry_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = DocumentRegistry(Path(temp_dir) / "registry.sqlite3")
            user_store = UserStore(Path(temp_dir) / "users.sqlite3")
            user_a = user_store.register(email="a@example.com", password="123456", full_name="User A")
            user_b = user_store.register(email="b@example.com", password="123456", full_name="User B")
            registry.create_job(
                doc_id="doc-a", owner_user_id=user_a.user_id, content_sha256="hash-a", job_id="job-a",
                file_name="a.pdf", file_type="pdf", size=1, source_path="a.pdf",
            )
            registry.create_job(
                doc_id="doc-b", owner_user_id=user_b.user_id, content_sha256="hash-b", job_id="job-b",
                file_name="b.pdf", file_type="pdf", size=1, source_path="b.pdf",
            )
            app.dependency_overrides[get_user_store] = lambda: user_store
            with patch("api.routes.get_document_registry", return_value=registry):
                client = TestClient(app)
                app.dependency_overrides[get_current_user] = lambda: user_a
                self.assertEqual([row["id"] for row in client.get("/documents").json()], ["doc-a"])
                app.dependency_overrides[get_current_user] = lambda: user_b
                self.assertEqual([row["id"] for row in client.get("/documents").json()], ["doc-b"])
            app.dependency_overrides.clear()
