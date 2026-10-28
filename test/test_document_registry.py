import tempfile
from pathlib import Path
import unittest

from api.auth import UserStore
from src.ingestion.document_registry import DocumentRegistry


class TestDocumentRegistry(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "rag.sqlite3"
        self.registry = DocumentRegistry(self.db_path)
        self.users = UserStore(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tracks_job_lifecycle_and_rejects_ready_duplicate(self):
        record = self.registry.create_job(
            doc_id="content-hash",
            job_id="job-1",
            file_name="policy.docx",
            file_type="docx",
            size=123,
            source_path="data/raw/uploads/content-hash.docx",
        )
        self.assertEqual(record["status"], "queued")

        self.registry.mark_processing("content-hash")
        self.registry.mark_ready(
            "content-hash",
            chunk_count=4,
            embedding_model="BAAI/bge-m3",
            collection_name="active-versioned",
            artifact_path="data/processed/documents/content-hash/chunks.jsonl",
        )
        ready = self.registry.get("content-hash")
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["chunk_count"], 4)

        with self.assertRaises(FileExistsError):
            self.registry.create_job(
                doc_id="content-hash",
                job_id="job-2",
                file_name="policy.docx",
                file_type="docx",
                size=123,
                source_path="data/raw/uploads/content-hash.docx",
            )

    def test_replace_requeues_same_document(self):
        self.registry.create_job(
            doc_id="content-hash",
            job_id="job-1",
            file_name="policy.docx",
            file_type="docx",
            size=123,
            source_path="source.docx",
        )
        self.registry.mark_ready(
            "content-hash",
            chunk_count=1,
            embedding_model="BAAI/bge-m3",
            collection_name="active-versioned",
            artifact_path="artifact.jsonl",
        )

        record = self.registry.create_job(
            doc_id="content-hash",
            job_id="job-2",
            file_name="policy.docx",
            file_type="docx",
            size=124,
            source_path="replacement.docx",
            replace_existing=True,
        )
        self.assertEqual(record["job_id"], "job-2")
        self.assertEqual(record["status"], "queued")
        self.assertEqual(record["chunk_count"], 0)

    def test_owner_scope_keeps_identical_content_in_separate_document_libraries(self):
        user_a = self.users.register(email="a@example.com", password="123456", full_name="User A")
        user_b = self.users.register(email="b@example.com", password="123456", full_name="User B")
        self.registry.create_job(
            doc_id="document-user-a",
            owner_user_id=user_a.user_id,
            content_sha256="same-content",
            job_id="job-a",
            file_name="policy.pdf",
            file_type="pdf",
            size=100,
            source_path="a.pdf",
        )
        self.registry.create_job(
            doc_id="document-user-b",
            owner_user_id=user_b.user_id,
            content_sha256="same-content",
            job_id="job-b",
            file_name="policy.pdf",
            file_type="pdf",
            size=100,
            source_path="b.pdf",
        )

        self.assertEqual(
            [record["doc_id"] for record in self.registry.list_documents(user_a.user_id)],
            ["document-user-a"],
        )
        self.assertEqual(
            self.registry.find_by_owner_content(user_b.user_id, "same-content")["doc_id"],
            "document-user-b",
        )
        self.assertIsNone(self.registry.get_owned("document-user-a", user_b.user_id))

    def test_document_page_is_owner_scoped_and_filters_names(self):
        user = self.users.register(email="page@example.com", password="123456", full_name="Page User")
        for index, name in enumerate(("alpha.pdf", "beta.pdf", "alpha-notes.docx")):
            self.registry.create_job(
                doc_id=f"doc-{index}", owner_user_id=user.user_id, content_sha256=f"hash-{index}",
                job_id=f"job-{index}", file_name=name, file_type="pdf", size=1, source_path=name,
            )
        items, total = self.registry.list_documents_page(
            user.user_id, page=1, page_size=1, query="alpha"
        )
        self.assertEqual(total, 2)
        self.assertEqual(len(items), 1)
        self.assertIn("alpha", items[0]["file_name"])


if __name__ == "__main__":
    unittest.main()
