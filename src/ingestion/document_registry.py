"""Persistent registry for upload/ingestion jobs.

The registry intentionally keeps operational state outside Qdrant.  Qdrant is
the retrieval store; SQLite records which uploaded file owns which points and
lets the UI poll progress without scanning the vector collection.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.storage.rag_database import RagDatabase


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentRegistry:
    """Document registry in the shared SQLite database."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.database = RagDatabase(db_path)
        self.db_path = self.database.db_path

    def _connection(self):
        return self.database.connection()

    @staticmethod
    def _as_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row else None

    def get(self, doc_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._as_dict(row)

    def get_owned(self, doc_id: str, owner_user_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE doc_id = ? AND owner_user_id = ?",
                (doc_id, owner_user_id),
            ).fetchone()
        return self._as_dict(row)

    def find_by_owner_content(self, owner_user_id: str, content_sha256: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE owner_user_id = ? AND content_sha256 = ?",
                (owner_user_id, content_sha256),
            ).fetchone()
        return self._as_dict(row)

    def list_documents(self, owner_user_id: str | None = None) -> list[dict[str, Any]]:
        with self._connection() as connection:
            if owner_user_id is None:
                rows = connection.execute(
                    "SELECT * FROM documents ORDER BY created_at DESC, file_name ASC"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM documents WHERE owner_user_id = ? ORDER BY created_at DESC, file_name ASC",
                    (owner_user_id,),
                ).fetchall()
        return [dict(row) for row in rows]

    def list_documents_page(
        self,
        owner_user_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
        query: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return one small owner-scoped page for the Streamlit document grid."""
        page = max(1, page)
        page_size = min(max(1, page_size), 100)
        filters = ["owner_user_id = ?"]
        values: list[Any] = [owner_user_id]
        if query and query.strip():
            filters.append("file_name LIKE ? ESCAPE '\\'")
            escaped = query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.append(f"%{escaped}%")
        where = " AND ".join(filters)
        with self._connection() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM documents WHERE {where}", values
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT * FROM documents
                WHERE {where}
                ORDER BY created_at DESC, file_name ASC
                LIMIT ? OFFSET ?
                """,
                (*values, page_size, (page - 1) * page_size),
            ).fetchall()
        return [dict(row) for row in rows], int(total)

    def count_active_jobs(self, owner_user_id: str) -> int:
        with self._connection() as connection:
            return int(connection.execute(
                """
                SELECT COUNT(*) FROM documents
                WHERE owner_user_id = ? AND status IN ('queued', 'processing', 'deleting')
                """,
                (owner_user_id,),
            ).fetchone()[0])

    def create_job(
        self,
        *,
        doc_id: str,
        job_id: str,
        file_name: str,
        file_type: str,
        size: int,
        source_path: str,
        owner_user_id: str | None = None,
        content_sha256: str | None = None,
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
            "owner_user_id": owner_user_id,
            "content_sha256": content_sha256 or doc_id,
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
                    doc_id, owner_user_id, content_sha256, job_id, file_name, file_type, size, status, source_path,
                    chunk_count, embedding_model, collection_name, artifact_path,
                    error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL, NULL, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    owner_user_id=excluded.owner_user_id,
                    content_sha256=excluded.content_sha256,
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
                    record["doc_id"], record["owner_user_id"], record["content_sha256"], record["job_id"],
                    record["file_name"], record["file_type"], record["size"], record["status"], record["source_path"],
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

    def mark_deleting(self, doc_id: str) -> None:
        self._update(doc_id, status="deleting", error_message=None)

    def remove(self, doc_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def assign_owner(self, doc_id: str, owner_user_id: str) -> dict[str, Any]:
        """Explicitly adopt one legacy document; never overwrite another owner."""
        with self._connection() as connection:
            result = connection.execute(
                """
                UPDATE documents
                SET owner_user_id = ?, content_sha256 = COALESCE(content_sha256, doc_id), updated_at = ?
                WHERE doc_id = ? AND (owner_user_id IS NULL OR owner_user_id = ?)
                """,
                (owner_user_id, _utc_now(), doc_id, owner_user_id),
            )
        if result.rowcount != 1:
            raise PermissionError("Tài liệu không tồn tại hoặc đã thuộc về user khác.")
        return self.get_owned(doc_id, owner_user_id) or {}

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
