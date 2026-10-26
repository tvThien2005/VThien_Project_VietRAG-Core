"""Privacy-preserving query telemetry stored in the shared RAG database."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import fmean
from typing import Any
from uuid import uuid4

from src.storage.rag_database import RagDatabase


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class QueryTelemetryStore:
    """Persist aggregate-safe operational facts, never questions, answers, or chunk text."""

    def __init__(self, database: RagDatabase | None = None):
        self.database = database or RagDatabase()

    def record(
        self,
        *,
        owner_user_id: str,
        status: str,
        search_mode: str,
        rerank_enabled: bool,
        top_k: int,
        top_n: int,
        source_count: int,
        timings_ms: dict[str, Any] | None = None,
        collection_name: str | None = None,
        llm_provider: str | None = None,
        error_type: str | None = None,
    ) -> None:
        timings = timings_ms or {}

        def duration(name: str) -> float:
            try:
                return max(0.0, float(timings.get(name, 0.0) or 0.0))
            except (TypeError, ValueError):
                return 0.0

        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO query_events (
                    event_id, owner_user_id, created_at, status, search_mode, rerank_enabled,
                    top_k, top_n, source_count, total_ms, retrieval_ms, embedding_ms, qdrant_search_ms,
                    bm25_search_ms, fusion_ms, rerank_ms, context_expansion_ms, generation_ms,
                    collection_name, llm_provider, error_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex, owner_user_id, _utc_now(), status, search_mode, int(rerank_enabled),
                    top_k, top_n, source_count, duration("total_ms"), duration("retrieval_ms"),
                    duration("embedding_ms"), duration("qdrant_search_ms"), duration("bm25_search_ms"),
                    duration("fusion_ms"), duration("rerank_ms"), duration("context_expansion_ms"),
                    duration("generation_ms"), collection_name,
                    llm_provider, error_type,
                ),
            )

    def dashboard(self, owner_user_id: str, *, days: int = 7) -> dict[str, list[dict[str, Any]]]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self.database.connection() as connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM query_events WHERE owner_user_id = ? AND created_at >= ? ORDER BY created_at DESC",
                    (owner_user_id, since),
                ).fetchall()
            ]

        count = len(rows)
        successful = [row for row in rows if row["status"] == "success"]
        contextual = [row for row in successful if row["source_count"] > 0]
        failures = [row for row in rows if row["status"] == "error"]

        def average(key: str, applicable_rows: list[dict[str, Any]] | None = None) -> float:
            values = applicable_rows if applicable_rows is not None else rows
            return fmean(float(row.get(key) or 0.0) for row in values) if values else 0.0

        avg_total = average("total_ms")
        hybrid_rows = [row for row in rows if row["search_mode"] == "hybrid"]
        reranked_rows = [row for row in rows if row["rerank_enabled"]]
        stage_rows = [
            ("Query embedding", "embedding_ms", "purple", None),
            ("Qdrant retrieval", "qdrant_search_ms", "blue", None),
            ("BM25 retrieval", "bm25_search_ms", "blue", hybrid_rows),
            ("Hybrid fusion (RRF)", "fusion_ms", "purple", hybrid_rows),
            ("Reranking", "rerank_ms", "amber", reranked_rows),
            ("Context expansion", "context_expansion_ms", "purple", None),
            ("LLM generation", "generation_ms", "emerald", None),
        ]
        latency = []
        for label, key, color, applicable_rows in stage_rows:
            value = average(key, applicable_rows)
            ratio = (value / avg_total) * 100 if avg_total else 0.0
            percentage = round(ratio)
            duration_text = f"{value:.0f}ms" if value >= 10 else f"{value:.2f}ms"
            percentage_text = f"{percentage}%" if percentage else ("<1%" if value > 0 else "0%")
            latency.append(
                {
                    "label": label,
                    "text": f"{duration_text} ({percentage_text})",
                    "pct": min(100, max(1, percentage) if value > 0 else 0),
                    "color": color,
                }
            )

        context_rate = (len(contextual) / len(successful) * 100) if successful else 0.0
        error_rate = (len(failures) / count * 100) if count else 0.0
        return {
            "kpis": [
                {"label": f"Truy vấn {days} ngày", "value": str(count), "delta": None, "tone": "indigo", "delta_tone": "indigo", "desc": "Sự kiện query của riêng bạn"},
                {"label": "Tỷ lệ có ngữ cảnh", "value": f"{context_rate:.1f}%", "delta": None, "tone": "emerald", "delta_tone": "emerald", "desc": "Query thành công có ít nhất một chunk"},
                {"label": "Độ trễ trung bình", "value": f"{avg_total:.0f}ms", "delta": None, "tone": "slate", "delta_tone": "slate", "desc": "End-to-end, gồm retrieval và LLM"},
                {"label": "Tỷ lệ lỗi", "value": f"{error_rate:.1f}%", "delta": None, "tone": "amber", "delta_tone": "amber", "desc": "Lỗi kết nối hoặc xử lý trong API"},
            ],
            "latency": latency,
        }
