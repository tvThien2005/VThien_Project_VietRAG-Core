"""Shared SQLite database for authentication and document ownership.

The migration is intentionally copy-only: historical ``users.sqlite3`` and
``document_registry.sqlite3`` are never modified or deleted.  It is safe to
construct this class repeatedly because every migration statement is
idempotent.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.config import PROJECT_ROOT


class RagDatabase:
    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        legacy_users_path: Path | str | None = None,
        legacy_documents_path: Path | str | None = None,
    ):
        self.db_path = Path(db_path or PROJECT_ROOT / "data" / "processed" / "rag.sqlite3")
        self._migrate_legacy = db_path is None or legacy_users_path is not None or legacy_documents_path is not None
        self.legacy_users_path = Path(legacy_users_path or PROJECT_ROOT / "data" / "processed" / "users.sqlite3")
        self.legacy_documents_path = Path(
            legacy_documents_path or PROJECT_ROOT / "data" / "processed" / "document_registry.sqlite3"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    full_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    owner_user_id TEXT REFERENCES users(user_id) ON DELETE RESTRICT,
                    content_sha256 TEXT,
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
            self._ensure_document_columns(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_owner_created "
                "ON documents(owner_user_id, created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_owner_content "
                "ON documents(owner_user_id, content_sha256)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS query_events (
                    event_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    search_mode TEXT NOT NULL,
                    rerank_enabled INTEGER NOT NULL,
                    top_k INTEGER NOT NULL,
                    top_n INTEGER NOT NULL,
                    source_count INTEGER NOT NULL,
                    total_ms REAL NOT NULL,
                    retrieval_ms REAL NOT NULL DEFAULT 0,
                    embedding_ms REAL NOT NULL DEFAULT 0,
                    qdrant_search_ms REAL NOT NULL DEFAULT 0,
                    bm25_search_ms REAL NOT NULL DEFAULT 0,
                    fusion_ms REAL NOT NULL DEFAULT 0,
                    rerank_ms REAL NOT NULL DEFAULT 0,
                    context_expansion_ms REAL NOT NULL DEFAULT 0,
                    generation_ms REAL NOT NULL DEFAULT 0,
                    collection_name TEXT,
                    llm_provider TEXT,
                    error_type TEXT
                )
                """
            )
            self._ensure_query_event_columns(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_query_events_owner_created "
                "ON query_events(owner_user_id, created_at DESC)"
            )
            if self._migrate_legacy:
                self._copy_legacy_users(connection)
                self._copy_legacy_documents(connection)

    @staticmethod
    def _ensure_document_columns(connection: sqlite3.Connection) -> None:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(documents)")}
        if "owner_user_id" not in columns:
            connection.execute("ALTER TABLE documents ADD COLUMN owner_user_id TEXT")
        if "content_sha256" not in columns:
            connection.execute("ALTER TABLE documents ADD COLUMN content_sha256 TEXT")

    @staticmethod
    def _ensure_query_event_columns(connection: sqlite3.Connection) -> None:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(query_events)")}
        for name in ("retrieval_ms", "fusion_ms"):
            if name not in columns:
                connection.execute(f"ALTER TABLE query_events ADD COLUMN {name} REAL NOT NULL DEFAULT 0")

    @staticmethod
    def _legacy_rows(path: Path, table_name: str) -> list[sqlite3.Row]:
        if not path.exists():
            return []
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        try:
            tables = {
                row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            if table_name not in tables:
                return []
            return connection.execute(f"SELECT * FROM {table_name}").fetchall()
        finally:
            connection.close()

    def _copy_legacy_users(self, connection: sqlite3.Connection) -> None:
        if self.legacy_users_path.resolve() == self.db_path.resolve():
            return
        for row in self._legacy_rows(self.legacy_users_path, "users"):
            values = dict(row)
            if not all(values.get(field) is not None for field in ("user_id", "email", "full_name", "password_hash")):
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO users (user_id, email, full_name, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    values["user_id"],
                    values["email"],
                    values["full_name"],
                    values["password_hash"],
                    values.get("created_at") or 0,
                ),
            )

    def _copy_legacy_documents(self, connection: sqlite3.Connection) -> None:
        if self.legacy_documents_path.resolve() == self.db_path.resolve():
            return
        user_ids = {row["user_id"] for row in connection.execute("SELECT user_id FROM users")}
        for row in self._legacy_rows(self.legacy_documents_path, "documents"):
            values = dict(row)
            required = ("doc_id", "job_id", "file_name", "file_type", "size", "status", "source_path", "created_at", "updated_at")
            if not all(values.get(field) is not None for field in required):
                continue
            owner_user_id = values.get("owner_user_id")
            # A missing/deleted historical account must not gain access by
            # accident.  The document stays unowned until explicitly adopted.
            if owner_user_id not in user_ids:
                owner_user_id = None
            connection.execute(
                """
                INSERT OR IGNORE INTO documents (
                    doc_id, owner_user_id, content_sha256, job_id, file_name, file_type, size, status,
                    chunk_count, embedding_model, collection_name, source_path, artifact_path,
                    error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    values["doc_id"], owner_user_id, values.get("content_sha256"), values["job_id"],
                    values["file_name"], values["file_type"], values["size"], values["status"],
                    values.get("chunk_count", 0), values.get("embedding_model"), values.get("collection_name"),
                    values["source_path"], values.get("artifact_path"), values.get("error_message"),
                    values["created_at"], values["updated_at"],
                ),
            )
