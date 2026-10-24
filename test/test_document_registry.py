import tempfile
from pathlib import Path
import unittest

from src.ingestion.document_registry import DocumentRegistry


class TestDocumentRegistry(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = DocumentRegistry(Path(self.temp_dir.name) / "registry.sqlite3")

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


if __name__ == "__main__":
    unittest.main()
