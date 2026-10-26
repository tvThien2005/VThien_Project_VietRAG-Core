import sqlite3
import tempfile
from pathlib import Path
from unittest import TestCase

from api.auth import UserStore
from src.ingestion.document_registry import DocumentRegistry
from src.storage.rag_database import RagDatabase


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
