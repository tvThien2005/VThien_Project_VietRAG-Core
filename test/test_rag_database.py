import sqlite3
import tempfile
from pathlib import Path
from unittest import TestCase

from api.auth import UserStore
from src.ingestion.document_registry import DocumentRegistry
from src.monitoring.telemetry import QueryTelemetryStore
from src.storage.rag_database import RagDatabase
from src.chat.history_store import ChatHistoryStore


class TestRagDatabase(TestCase):
    def test_copy_only_migration_keeps_accounts_and_links_documents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_users = root / "users.sqlite3"
            legacy_documents = root / "document_registry.sqlite3"
            unified = root / "rag.sqlite3"

            legacy_user_store = UserStore(legacy_users)
            user = legacy_user_store.register(
                email="demo@example.com", password="123456", full_name="Demo User"
            )

            # The prior document registry has no foreign-key constraint.  Use
            # its historical shape here to verify that migration adds the link.
            connection = sqlite3.connect(legacy_documents)
            try:
                connection.execute(
                    """
                    CREATE TABLE documents (
                        doc_id TEXT PRIMARY KEY, owner_user_id TEXT, content_sha256 TEXT,
                        job_id TEXT NOT NULL, file_name TEXT NOT NULL, file_type TEXT NOT NULL,
                        size INTEGER NOT NULL, status TEXT NOT NULL, chunk_count INTEGER NOT NULL,
                        embedding_model TEXT, collection_name TEXT, source_path TEXT NOT NULL,
                        artifact_path TEXT, error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "doc-1", user.user_id, "sha", "job-1", "demo.pdf", "pdf", 42, "ready", 2,
                        "model", "collection", "source.pdf", "chunks.jsonl", None,
                        "2026-01-01T00:00:00+00:00", "2026-01-01T00:01:00+00:00",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            RagDatabase(
                unified,
                legacy_users_path=legacy_users,
                legacy_documents_path=legacy_documents,
            )
            migrated_user_store = UserStore(unified)
            migrated_registry = DocumentRegistry(unified)

            self.assertEqual(
                migrated_user_store.authenticate(email="demo@example.com", password="123456"), user
            )
            self.assertEqual(migrated_registry.get_owned("doc-1", user.user_id)["file_name"], "demo.pdf")
            self.assertTrue(legacy_users.exists())
            self.assertTrue(legacy_documents.exists())

            connection = sqlite3.connect(unified)
            try:
                foreign_keys = connection.execute("PRAGMA foreign_key_list(documents)").fetchall()
            finally:
                connection.close()
            self.assertEqual(foreign_keys[0][2:5], ("users", "owner_user_id", "user_id"))

    def test_registry_allows_legacy_ownerless_records_but_rejects_unknown_owner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "rag.sqlite3"
            registry = DocumentRegistry(db_path)
            registry.create_job(
                doc_id="legacy-doc", job_id="job-1", file_name="legacy.pdf", file_type="pdf",
                size=1, source_path="legacy.pdf",
            )
            self.assertIsNone(registry.get("legacy-doc")["owner_user_id"])

            with self.assertRaises(sqlite3.IntegrityError):
                registry.create_job(
                    doc_id="invalid-owner", owner_user_id="not-a-user", job_id="job-2",
                    file_name="invalid.pdf", file_type="pdf", size=1, source_path="invalid.pdf",
                )

    def test_query_telemetry_is_owner_scoped_and_has_stage_averages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = RagDatabase(Path(temp_dir) / "rag.sqlite3")
            users = UserStore(database.db_path)
            user_a = users.register(email="a@example.com", password="123456", full_name="User A")
            user_b = users.register(email="b@example.com", password="123456", full_name="User B")
            telemetry = QueryTelemetryStore(database)
            telemetry.record(
                owner_user_id=user_a.user_id, status="success", search_mode="hybrid", rerank_enabled=False,
                top_k=10, top_n=4, source_count=2,
                timings_ms={
                    "total_ms": 200, "embedding_ms": 20, "qdrant_search_ms": 30,
                    "bm25_search_ms": 10, "fusion_ms": 0.1, "generation_ms": 150,
                },
            )
            telemetry.record(
                owner_user_id=user_b.user_id, status="error", search_mode="vector", rerank_enabled=False,
                top_k=10, top_n=4, source_count=0, timings_ms={"total_ms": 50}, error_type="ConnectionError",
            )

            dashboard = telemetry.dashboard(user_a.user_id)
            self.assertEqual(dashboard["kpis"][0]["value"], "1")
            self.assertEqual(dashboard["kpis"][1]["value"], "100.0%")
            self.assertEqual(dashboard["kpis"][3]["value"], "0.0%")
            self.assertIn("20ms", dashboard["latency"][0]["text"])
            self.assertIn("0.10ms", dashboard["latency"][3]["text"])

    def test_history_keeps_compact_sources_and_cascades_session_deletion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = RagDatabase(Path(temp_dir) / "rag.sqlite3")
            users = UserStore(database.db_path)
            user = users.register(email="history@example.com", password="123456", full_name="History User")
            history = ChatHistoryStore(database)
            session_id, user_message_id = history.create_user_message(
                owner_user_id=user.user_id, question="<script>alert(1)</script>", session_id=None,
            )
            assistant_message_id = history.complete_response(
                owner_user_id=user.user_id, session_id=session_id, user_message_id=user_message_id,
                answer="Câu trả lời", sources=[{
                    "chunk_id": "chunk-1", "doc_name": "manual.pdf", "section_path": ["Chương 2", "Mục 2.1"],
                    "page": 4, "score": 0.91, "text": "Không được lưu lại trong SQLite",
                }],
            )
            history.save_feedback(
                owner_user_id=user.user_id, message_id=assistant_message_id, rating="up", comment="", category="other",
            )
            loaded = history.get_session_messages(owner_user_id=user.user_id, session_id=session_id)
            self.assertEqual(len(loaded["messages"]), 2)
            source = loaded["messages"][1]["sources"][0]
            self.assertEqual(source["section_path"], "Chương 2 > Mục 2.1")
            self.assertNotIn("text", source)

            self.assertTrue(history.delete_session(owner_user_id=user.user_id, session_id=session_id))
            with database.connection() as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0], 0)

    def test_history_is_owner_scoped_and_expired_sessions_are_purged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = RagDatabase(Path(temp_dir) / "rag.sqlite3")
            users = UserStore(database.db_path)
            owner = users.register(email="owner@example.com", password="123456", full_name="Owner")
            other = users.register(email="other@example.com", password="123456", full_name="Other")
            history = ChatHistoryStore(database)
            session_id, _ = history.create_user_message(
                owner_user_id=owner.user_id, question="Question", session_id=None,
            )
            self.assertEqual(history.list_sessions(owner_user_id=other.user_id)["items"], [])
            self.assertFalse(history.delete_session(owner_user_id=other.user_id, session_id=session_id))
            with database.connection() as connection:
                connection.execute(
                    "UPDATE chat_sessions SET updated_at = '2000-01-01T00:00:00+00:00' WHERE session_id = ?",
                    (session_id,),
                )
            self.assertEqual(history.purge_expired(), 1)
            self.assertEqual(history.list_sessions(owner_user_id=owner.user_id)["items"], [])
