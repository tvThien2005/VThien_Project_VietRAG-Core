"""SQLite storage for chat history and per-answer feedback.

Only the user/assistant text and a compact citation reference are persisted.
The retrieved chunk body is deliberately excluded: it remains in Qdrant and
is returned only for the live answer.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from src.storage.rag_database import RagDatabase


class ChatHistoryStore:
    RETENTION_DAYS = 90
    DEFAULT_SESSION_PAGE_SIZE = 30
    DEFAULT_MESSAGE_PAGE_SIZE = 100

    def __init__(self, database: RagDatabase | None = None):
        self.database = database or RagDatabase()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @classmethod
    def _retention_cutoff(cls) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=cls.RETENTION_DAYS)).isoformat(timespec="seconds")

    def purge_expired(self) -> int:
        """Delete inactive sessions older than the retention period.

        Foreign-key cascades remove their messages and feedback in the same
        transaction.  This runs on history writes/reads; no scheduler is
        required for a local deployment.
        """
        with self.database.connection() as connection:
            cursor = connection.execute(
                "DELETE FROM chat_sessions WHERE updated_at < ?", (self._retention_cutoff(),)
            )
            return cursor.rowcount

    @staticmethod
    def _compact_sources(sources: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Keep citation metadata only; never duplicate retrieved chunk text."""
        compact: list[dict[str, Any]] = []
        for source in sources or []:
            if not isinstance(source, dict):
                continue
            score = source.get("score")
            try:
                score = float(score) if score is not None else None
            except (TypeError, ValueError):
                score = None
            section_path = source.get("section_path")
            if isinstance(section_path, list):
                section_path = " > ".join(str(item) for item in section_path if str(item).strip())
            compact.append(
                {
                    "chunk_id": str(source.get("chunk_id") or ""),
                    "doc_name": str(source.get("doc_name") or "Không rõ tài liệu"),
                    "section_path": str(section_path or ""),
                    "page": source.get("page"),
                    "score": score,
                }
            )
        return compact

    @staticmethod
    def _safe_json(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode_json(value: str | None, fallback: Any) -> Any:
        if not value:
            return fallback
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return fallback

    def create_user_message(self, *, owner_user_id: str, question: str, session_id: str | None) -> tuple[str, str]:
        """Persist the question before generation so failures remain auditable."""
        self.purge_expired()
        now = self._now()
        session_id = session_id or uuid4().hex
        message_id = uuid4().hex
        title = " ".join(question.split())[:120] or "Cuộc trò chuyện"
        with self.database.connection() as connection:
            existing = connection.execute(
                "SELECT session_id FROM chat_sessions WHERE session_id = ? AND owner_user_id = ?",
                (session_id, owner_user_id),
            ).fetchone()
            if existing is None:
                # Do not allow an arbitrary client-provided id to create a
                # session if it already belongs to another user.
                collision = connection.execute(
                    "SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                if collision is not None:
                    raise PermissionError("Không tìm thấy cuộc trò chuyện.")
                connection.execute(
                    """
                    INSERT INTO chat_sessions (session_id, owner_user_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, owner_user_id, title, now, now),
                )
            connection.execute(
                """
                INSERT INTO chat_messages (
                    message_id, session_id, owner_user_id, role, content, status, created_at
                ) VALUES (?, ?, ?, 'user', ?, 'pending', ?)
                """,
                (message_id, session_id, owner_user_id, question, now),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ? AND owner_user_id = ?",
                (now, session_id, owner_user_id),
            )
        return session_id, message_id

    def complete_response(
        self,
        *,
        owner_user_id: str,
        session_id: str,
        user_message_id: str,
        answer: str,
        sources: list[dict[str, Any]] | None,
    ) -> str:
        """Atomically mark the question complete and add one complete answer."""
        now = self._now()
        assistant_message_id = uuid4().hex
        with self.database.connection() as connection:
            user_message = connection.execute(
                """
                SELECT message_id FROM chat_messages
                WHERE message_id = ? AND session_id = ? AND owner_user_id = ? AND role = 'user'
                """,
                (user_message_id, session_id, owner_user_id),
            ).fetchone()
            if user_message is None:
                raise PermissionError("Không tìm thấy câu hỏi thuộc phiên chat này.")
            connection.execute(
                """
                INSERT INTO chat_messages (
                    message_id, session_id, owner_user_id, role, content, status,
                    sources_json, created_at, completed_at
                ) VALUES (?, ?, ?, 'assistant', ?, 'complete', ?, ?, ?)
                """,
                (
                    assistant_message_id, session_id, owner_user_id, answer,
                    self._safe_json(self._compact_sources(sources)), now, now,
                ),
            )
            connection.execute(
                """
                UPDATE chat_messages SET status = 'complete', completed_at = ?, error_message = NULL
                WHERE message_id = ? AND owner_user_id = ?
                """,
                (now, user_message_id, owner_user_id),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ? AND owner_user_id = ?",
                (now, session_id, owner_user_id),
            )
        return assistant_message_id

    def mark_user_message_error(self, *, owner_user_id: str, message_id: str, error_message: str) -> None:
        """Leave a clear terminal status and never create a partial assistant row."""
        now = self._now()
        with self.database.connection() as connection:
            connection.execute(
                """
                UPDATE chat_messages
                SET status = 'error', error_message = ?, completed_at = ?
                WHERE message_id = ? AND owner_user_id = ? AND role = 'user' AND status = 'pending'
                """,
                (error_message[:500], now, message_id, owner_user_id),
            )

    def list_sessions(self, *, owner_user_id: str, page: int = 1, page_size: int = DEFAULT_SESSION_PAGE_SIZE) -> dict:
        self.purge_expired()
        page = max(1, page)
        page_size = min(max(1, page_size), 100)
        offset = (page - 1) * page_size
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT session_id, title, updated_at
                FROM chat_sessions
                WHERE owner_user_id = ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (owner_user_id, page_size + 1, offset),
            ).fetchall()
        has_more = len(rows) > page_size
        return {
            "items": [dict(row) for row in rows[:page_size]],
            "page": page,
            "page_size": page_size,
            "has_more": has_more,
        }

    def get_session_messages(
        self, *, owner_user_id: str, session_id: str, before_created_at: str | None = None,
        limit: int = DEFAULT_MESSAGE_PAGE_SIZE,
    ) -> dict:
        self.purge_expired()
        limit = min(max(1, limit), 100)
        with self.database.connection() as connection:
            owned = connection.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id = ? AND owner_user_id = ?",
                (session_id, owner_user_id),
            ).fetchone()
            if owned is None:
                raise KeyError(session_id)
            clauses = ["session_id = ?", "owner_user_id = ?"]
            values: list[Any] = [session_id, owner_user_id]
            if before_created_at:
                clauses.append("created_at < ?")
                values.append(before_created_at)
            rows = connection.execute(
                f"""
                SELECT message_id, role, content, status, error_message, sources_json, created_at
                FROM chat_messages
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (*values, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        selected = list(reversed(rows[:limit]))
        messages = []
        for row in selected:
            item = {
                "message_id": row["message_id"], "role": row["role"], "content": row["content"],
                "status": row["status"], "error": row["error_message"], "ts": row["created_at"],
                "sources": self._decode_json(row["sources_json"], []),
            }
            messages.append(item)
        next_before = rows[limit - 1]["created_at"] if has_more and len(rows) > limit else None
        return {"messages": messages, "has_more": has_more, "next_before_created_at": next_before}

    def delete_session(self, *, owner_user_id: str, session_id: str) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                "DELETE FROM chat_sessions WHERE session_id = ? AND owner_user_id = ?",
                (session_id, owner_user_id),
            )
            return cursor.rowcount > 0

    def save_feedback(
        self, *, owner_user_id: str, message_id: str, rating: str, comment: str, category: str,
    ) -> None:
        now = self._now()
        with self.database.connection() as connection:
            message = connection.execute(
                """
                SELECT message_id FROM chat_messages
                WHERE message_id = ? AND owner_user_id = ? AND role = 'assistant' AND status = 'complete'
                """,
                (message_id, owner_user_id),
            ).fetchone()
            if message is None:
                raise KeyError(message_id)
            connection.execute(
                """
                INSERT INTO feedback (
                    feedback_id, owner_user_id, message_id, rating, comment, category, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_user_id, message_id) DO UPDATE SET
                    rating = excluded.rating, comment = excluded.comment,
                    category = excluded.category, updated_at = excluded.updated_at
                """,
                (uuid4().hex, owner_user_id, message_id, rating, comment, category, now, now),
            )
