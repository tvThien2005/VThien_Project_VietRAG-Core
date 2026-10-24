"""Persistent registry for upload/ingestion jobs.

The registry intentionally keeps operational state outside Qdrant.  Qdrant is
the retrieval store; SQLite records which uploaded file owns which points and
lets the UI poll progress without scanning the vector collection.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import PROJECT_ROOT


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentRegistry:
    """Small SQLite registry; each operation opens its own connection for job safety."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path or PROJECT_ROOT / "data" / "processed" / "document_registry.sqlite3")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    embedding_model TEXT,
                    collection_name TEXT,
                    source_path TEXT NOT NULL,
                    artifact_path TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _as_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row else None

    def get(self, doc_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._as_dict(row)

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, file_name ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def create_job(
        self,
        *,
        doc_id: str,
        job_id: str,
        file_name: str,
        file_type: str,
        size: int,
        source_path: str,
        replace_existing: bool = False,
    ) -> dict[str, Any]:
        existing = self.get(doc_id)
        if existing and existing["status"] in {"queued", "processing"}:
            raise ValueError("Tài liệu này đang được xử lý.")
        if existing and existing["status"] == "ready" and not replace_existing:
            raise FileExistsError("Tài liệu đã được index. Hãy dùng replace để lập chỉ mục lại.")

        now = _utc_now()
        record = {
            "doc_id": doc_id,
            "job_id": job_id,
            "file_name": file_name,
            "file_type": file_type,
            "size": size,
            "status": "queued",
            "source_path": source_path,
            "created_at": existing["created_at"] if existing else now,
            "updated_at": now,
        }
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    doc_id, job_id, file_name, file_type, size, status, source_path,
                    chunk_count, embedding_model, collection_name, artifact_path,
                    error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL, NULL, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    job_id=excluded.job_id,
                    file_name=excluded.file_name,
                    file_type=excluded.file_type,
                    size=excluded.size,
                    status=excluded.status,
                    source_path=excluded.source_path,
                    chunk_count=0,
                    embedding_model=NULL,
                    collection_name=NULL,
                    artifact_path=NULL,
                    error_message=NULL,
                    updated_at=excluded.updated_at
                """,
                (
                    record["doc_id"], record["job_id"], record["file_name"], record["file_type"],
                    record["size"], record["status"], record["source_path"],
                    record["created_at"], record["updated_at"],
                ),
            )
        return self.get(doc_id) or record

    def mark_processing(self, doc_id: str) -> None:
        self._update(doc_id, status="processing", error_message=None)

    def mark_ready(
        self,
        doc_id: str,
        *,
        chunk_count: int,
        embedding_model: str,
        collection_name: str,
        artifact_path: str,
    ) -> None:
        self._update(
            doc_id,
            status="ready",
            chunk_count=chunk_count,
            embedding_model=embedding_model,
            collection_name=collection_name,
            artifact_path=artifact_path,
            error_message=None,
        )

    def mark_failed(self, doc_id: str, message: str) -> None:
        self._update(doc_id, status="failed", error_message=message[:2000])

    def remove(self, doc_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def _update(self, doc_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = _utc_now()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._connection() as connection:
            result = connection.execute(
                f"UPDATE documents SET {assignments} WHERE doc_id = ?",
                (*fields.values(), doc_id),
            )
        if result.rowcount != 1:
            raise KeyError(f"Document {doc_id!r} is not registered.")
