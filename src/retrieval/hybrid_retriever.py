
from time import perf_counter
from typing import Dict, List, Optional

from qdrant_client.http import models

from src.config import get_config
from src.logger import get_logger
from src.retrieval.base import BaseRetriever
from src.retrieval.qdrant_filters import build_payload_filter
from src.retrieval.vector_retriever import VectorRetriever
from src.schemas.response import RetrievedChunk

logger = get_logger("hybrid_retriever")


class HybridRetriever(BaseRetriever):
    """
    Hybrid Retrieval kết hợp:
    - Dense retrieval: Semantic search qua VectorRetriever.
    - Sparse retrieval: BM25 vector được Qdrant lập chỉ mục.
    - Fusion: Weighted Reciprocal Rank Fusion (RRF).

    Yêu cầu:
    - Collection có sparse vector tên `bm25`.
    - Các chunks đã được ingest với biểu diễn BM25 tương ứng.
    - Qdrant hỗ trợ inference BM25 và qdrant-client tương thích.
    """

    def __init__(
        self,
        vector_retriever: Optional[VectorRetriever] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        candidate_k: Optional[int] = None,
        candidate_multiplier: Optional[int] = None,
        min_candidates: Optional[int] = None,
        rrf_k: Optional[int] = None,
        bm25_vector_name: str = "bm25",
        bm25_model: str = "Qdrant/bm25",
        bm25_options: Optional[Dict] = None,
    ):
        self.cfg = get_config()
        self.vector_retriever = vector_retriever or VectorRetriever()

        hybrid_cfg = self.cfg.retrieval.hybrid_search

        self.dense_weight = float(
            dense_weight
            if dense_weight is not None
            else hybrid_cfg.dense_weight
        )
        self.sparse_weight = float(
            sparse_weight
            if sparse_weight is not None
            else hybrid_cfg.sparse_weight
        )

        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("Retrieval weights must be non-negative.")

        if self.dense_weight == 0 and self.sparse_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than zero."
            )

        self.candidate_k = candidate_k if candidate_k is not None else hybrid_cfg.candidate_k
        self.candidate_multiplier = max(1, candidate_multiplier if candidate_multiplier is not None else hybrid_cfg.candidate_multiplier)
        self.min_candidates = max(1, min_candidates if min_candidates is not None else hybrid_cfg.min_candidates)
        self.rrf_k = max(1, rrf_k if rrf_k is not None else hybrid_cfg.rrf_k)

        self.bm25_vector_name = bm25_vector_name
        self.bm25_model = bm25_model
        self.bm25_options = bm25_options or {}
    def _get_candidate_limit(self, final_k: int) -> int:
        """
        Lấy nhiều ứng viên hơn số kết quả cuối cùng để tăng recall.
        candidate_k được ưu tiên nếu được cấu hình.
        """
        if self.candidate_k is not None:
            return max(final_k, self.candidate_k)

        return max(
            final_k * self.candidate_multiplier,
            self.min_candidates,
        )

    def _retrieve_sparse(
        self,
        query: str,
        candidate_k: int,
        query_filter: Optional[models.Filter] = None,
    ) -> List[RetrievedChunk]:
        """
        BM25 retrieval trực tiếp trên sparse vector của Qdrant.
        Không scroll toàn collection và không tính BM25 bằng Python.
        """
        response = self.vector_retriever.client.query_points(
            collection_name=self.vector_retriever.collection_name,
            query=models.Document(
                text=query,
                model=self.bm25_model,
                options=self.bm25_options,  
            ),
            using=self.bm25_vector_name,
            limit=candidate_k,
            with_payload=True,
            with_vectors=False,
            query_filter=query_filter,
        )

        chunks: List[RetrievedChunk] = []

        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            text = payload.get("text", "")

            # Không đưa các điểm thiếu nội dung vào context RAG.
            if not text:
                continue

            chunks.append(
                RetrievedChunk(
                    chunk_id=str(point.id),
                    doc_id=payload.get("doc_id"),
                    text=text,
                    score=float(point.score or 0.0),
                    rank=rank,
                    metadata=dict(payload),
                )
            )

        return chunks

    def _fuse_results(
        self,
        dense_results: List[RetrievedChunk],
        sparse_results: List[RetrievedChunk],
        limit: int,
    ) -> List[RetrievedChunk]:
        """
        Weighted RRF:
            score(d) = sum(weight / (rrf_k + rank))

        Rank bắt đầu từ 1.
        Điểm cuối cùng là điểm fusion, không phải cosine similarity
        hay raw BM25 score.
        """
        chunk_map: Dict[str, RetrievedChunk] = {}
        fused_scores: Dict[str, float] = {}

        sources = (
            (dense_results, self.dense_weight),
            (sparse_results, self.sparse_weight),
        )

        for results, weight in sources:
            if weight == 0:
                continue

            # Thứ hạng của nguồn được xác định bởi thứ tự trả về.
            for rank, chunk in enumerate(results, start=1):
                chunk_id = str(chunk.chunk_id)

                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk

                fused_scores[chunk_id] = (
                    fused_scores.get(chunk_id, 0.0)
                    + weight / (self.rrf_k + rank)
                )

        ranked_ids = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        final_results: List[RetrievedChunk] = []

        for final_rank, (chunk_id, fusion_score) in enumerate(
            ranked_ids[:limit],
            start=1,
        ):
            chunk = chunk_map[chunk_id]

            final_results.append(
                chunk.model_copy(
                    update={
                        "score": float(fusion_score),
                        "rank": final_rank,
                    }
                )
            )

        return final_results

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        owner_user_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        file_name: Optional[str] = None,
        section_id: Optional[str] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[RetrievedChunk]:
        limit = (
            top_k
            if top_k is not None
            else self.cfg.retrieval.top_k
        )

        if limit <= 0 or not query or not query.strip():
            return []

        query = query.strip()
        candidate_k = self._get_candidate_limit(limit)
        query_filter = build_payload_filter(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            file_name=file_name,
            section_id=section_id,
        )

        logger.info(
            "Hybrid Search | query=%r | final_k=%d | candidate_k=%d "
            "| dense_w=%.2f | sparse_w=%.2f",
            query,
            limit,
            candidate_k,
            self.dense_weight,
            self.sparse_weight,
        )

        # 1. Dense retrieval.
        dense_results: List[RetrievedChunk] = []

        if self.dense_weight > 0:
            if query_filter is None:
                dense_results = self.vector_retriever.retrieve(
                    query=query,
                    top_k=candidate_k,
                    timings=timings,
                )
            else:
                dense_results = self.vector_retriever.retrieve(
                    query=query,
                    top_k=candidate_k,
                    owner_user_id=owner_user_id,
                    doc_id=doc_id,
                    file_name=file_name,
                    section_id=section_id,
                    timings=timings,
                )

        # 2. Sparse BM25 retrieval.
        sparse_results: List[RetrievedChunk] = []

        if self.sparse_weight > 0:
            try:
                started = perf_counter()
                sparse_results = self._retrieve_sparse(
                    query=query,
                    candidate_k=candidate_k,
                    query_filter=query_filter,
                )
                if timings is not None:
                    timings["bm25_search_ms"] = timings.get("bm25_search_ms", 0.0) + (perf_counter() - started) * 1000
            except Exception:
                logger.exception(
                    "BM25 retrieval thất bại. "
                    "Tiếp tục với các nguồn retrieval còn hoạt động."
                )

        # 3. Rank fusion.
        started = perf_counter()
        final_results = self._fuse_results(
            dense_results=dense_results,
            sparse_results=sparse_results,
            limit=limit,
        )
        if timings is not None:
            timings["fusion_ms"] = timings.get("fusion_ms", 0.0) + (perf_counter() - started) * 1000

        logger.info(
            "Hybrid Search hoàn tất | dense=%d | sparse=%d | final=%d",
            len(dense_results),
            len(sparse_results),
            len(final_results),
        )

        return final_results
