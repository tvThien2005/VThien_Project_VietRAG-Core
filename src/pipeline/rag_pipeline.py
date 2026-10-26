import time
from typing import Iterator, List, Optional, Tuple

from src.config import get_config
from src.generation.generator import Generator
from src.logger import get_logger
from src.retrieval.pipeline import RetrievalPipeline
from src.schemas.response import RAGResponse, RetrievedChunk

logger = get_logger("rag_pipeline")

MIN_CONTEXT_SCORE = 0.1
NO_CONTEXT_ANSWER = (
    "Dựa trên tài liệu được cung cấp, tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi này."
)


class RAGPipeline:
    """End-to-End RAG Pipeline: Kết hợp Retrieval, Reranking và LLM Generation."""
    def __init__(
        self,
        retrieval_pipeline: Optional[RetrievalPipeline] = None,
        generator: Optional[Generator] = None,
    ):
        self.cfg = get_config()
        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline()
        self.generator = generator or Generator()

    def _filter_low_score_chunks(
        self,
        chunks: List[RetrievedChunk],
        use_rerank: Optional[bool],
        use_hybrid: Optional[bool],
    ) -> List[RetrievedChunk]:
        rerank_enabled = (
            self.cfg.retrieval.reranker.enabled
            if use_rerank is None
            else use_rerank
        )
        hybrid_enabled = (
            self.cfg.retrieval.hybrid_search.enabled
            if use_hybrid is None
            else use_hybrid
        )
        if rerank_enabled or hybrid_enabled:
            return chunks
        return [chunk for chunk in chunks if chunk.score >= MIN_CONTEXT_SCORE]

    def query(
        self,
        query_text: Optional[str] = None,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        use_rerank: Optional[bool] = None,
        use_hybrid: Optional[bool] = None,
        owner_user_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        file_name: Optional[str] = None,
        section_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> RAGResponse:
        """Thực hiện chu trình RAG hoàn chỉnh: Truy xuất ngữ cảnh và sinh câu trả lời.
        
        - use_rerank: True/False để bật/tắt Reranker cho lần gọi này (hoặc None để theo config.yaml).
        - use_hybrid: True/False để bật/tắt Hybrid Search cho lần gọi này (hoặc None để theo config.yaml).
        """
        target_query = (query or query_text or "").strip()
        if not target_query:
            raise ValueError("Câu hỏi (query/query_text) không được rỗng.")

        start_time = time.time()
        timings_ms: dict[str, float] = {}
        logger.info(f"Bắt đầu chu trình RAG Pipeline cho câu hỏi: '{target_query}'")

        # 1. Truy xuất tài liệu phù hợp (Retrieval + Rerank)
        filters = {
            key: value
            for key, value in (
                ("owner_user_id", owner_user_id),
                ("doc_id", doc_id),
                ("file_name", file_name),
                ("section_id", section_id),
            )
            if value is not None
        }
        chunks = self.retrieval_pipeline.retrieve(
            query=target_query,
            top_k=top_k,
            top_n=top_n,
            use_rerank=use_rerank,
            use_hybrid=use_hybrid,
            timings=timings_ms,
            **filters,
        )
        chunks = self._filter_low_score_chunks(chunks, use_rerank, use_hybrid)

        # 2. Xử lý trường hợp không tìm thấy dữ liệu liên quan
        if not chunks:
            latency = time.time() - start_time
            timings_ms["total_ms"] = latency * 1000
            return RAGResponse(
                query=target_query,
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                latency_seconds=latency,
                metadata={"timings_ms": timings_ms},
            )

        # 3. Sinh câu trả lời qua LLM Generator
        generation_started = time.perf_counter()
        answer = self.generator.generate_answer(
            query=target_query,
            chunks=chunks,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        timings_ms["generation_ms"] = (time.perf_counter() - generation_started) * 1000

        latency = time.time() - start_time
        timings_ms["total_ms"] = latency * 1000
        logger.info(f"Hoàn tất xử lý RAG Pipeline trong {latency:.2f}s")

        return RAGResponse(
            query=target_query,
            answer=answer,
            sources=chunks,
            latency_seconds=latency,
            metadata={
                "retrieved_count": len(chunks),
                "llm_provider": self.cfg.llm.default_provider,
                "use_rerank": use_rerank,
                "use_hybrid": use_hybrid,
                "timings_ms": timings_ms,
            },
        )

    def stream_query(
        self,
        query_text: Optional[str] = None,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        use_rerank: Optional[bool] = None,
        use_hybrid: Optional[bool] = None,
        owner_user_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        file_name: Optional[str] = None,
        section_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[List[RetrievedChunk], Iterator[str]]:
        """Truy xuất tài liệu trước, sau đó stream câu trả lời từ LLM (thích hợp cho Web UI / SSE)."""
        target_query = (query or query_text or "").strip()
        if not target_query:
            raise ValueError("Câu hỏi (query/query_text) không được rỗng.")

        logger.info(f"Bắt đầu Stream RAG Pipeline cho câu hỏi: '{target_query}'")

        # 1. Truy xuất tài liệu
        filters = {
            key: value
            for key, value in (
                ("owner_user_id", owner_user_id),
                ("doc_id", doc_id),
                ("file_name", file_name),
                ("section_id", section_id),
            )
            if value is not None
        }
        chunks = self.retrieval_pipeline.retrieve(
            query=target_query,
            top_k=top_k,
            top_n=top_n,
            use_rerank=use_rerank,
            use_hybrid=use_hybrid,
            **filters,
        )
        chunks = self._filter_low_score_chunks(chunks, use_rerank, use_hybrid)

        if not chunks:
            def empty_stream():
                yield NO_CONTEXT_ANSWER

            return [], empty_stream()

        # 2. Tạo iterator stream từ generator
        stream_iter = self.generator.stream_answer(
            query=target_query,
            chunks=chunks,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return chunks, stream_iter


# Alias thuận tiện
RagPipeline = RAGPipeline
