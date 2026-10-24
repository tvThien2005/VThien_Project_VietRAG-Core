
from typing import List, Optional

from src.config import get_config
from src.logger import get_logger
from src.retrieval.context_expander import ContextExpander, create_context_tokenizer
from src.retrieval.base import BaseReranker, BaseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import get_reranker
from src.retrieval.vector_retriever import VectorRetriever
from src.schemas.response import RetrievedChunk

logger = get_logger("retrieval_pipeline")


class RetrievalPipeline:
    """
    Pipeline truy xuat:
    1. Dense Retrieval hoac Hybrid Retrieval.
    2. Reranking neu duoc bat.
    3. Tra ve top_n chunks cuoi cung.

    top_k: so chunks candidate lay tu retriever.
    top_n: so chunks cuoi cung sau pipeline.
    """

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        reranker: Optional[BaseReranker] = None,
        *,
        vector_retriever: Optional[VectorRetriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        context_expander: Optional[ContextExpander] = None,
    ):
        self.cfg = get_config(reload=True)

        # Cho phep tai su dung VectorRetriever da co.
        if vector_retriever is not None:
            self.vector_retriever = vector_retriever
        elif isinstance(retriever, VectorRetriever):
            self.vector_retriever = retriever
        else:
            self.vector_retriever = VectorRetriever()

        # Dense Retriever duoc chia se voi HybridRetriever.
        self.hybrid_retriever = (
            hybrid_retriever
            or HybridRetriever(
                vector_retriever=self.vector_retriever
            )
        )

        # Retriever dung khi use_hybrid=False.
        self.retriever = retriever or self.vector_retriever

        if reranker is not None:
            self.reranker = reranker
        elif self.cfg.retrieval.reranker.enabled:
            self.reranker = get_reranker()
        else:
            self.reranker = None

        expansion_cfg = self.cfg.retrieval.context_expansion
        if context_expander is not None:
            self.context_expander = context_expander
        elif expansion_cfg.enabled:
            self.context_expander = ContextExpander(
                client=self.vector_retriever.client,
                collection_name=self.vector_retriever.collection_name,
                tokenizer=create_context_tokenizer(),
                config=expansion_cfg,
            )
        else:
            self.context_expander = None

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        use_rerank: Optional[bool] = None,
        use_hybrid: Optional[bool] = None,
        doc_id: Optional[str] = None,
        file_name: Optional[str] = None,
        section_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:

        if not query or not query.strip():
            logger.warning("Cau truy van rong.")
            return []

        # Dung is None thay vi "or" de tranh bo qua gia tri 0.
        limit_k = (
            self.cfg.retrieval.top_k
            if top_k is None
            else top_k
        )
        limit_n = (
            self.cfg.retrieval.reranker.top_n
            if top_n is None
            else top_n
        )

        if limit_k <= 0:
            raise ValueError("top_k phai lon hon 0.")

        if limit_n <= 0:
            raise ValueError("top_n phai lon hon 0.")

        enable_hybrid = (
            use_hybrid
            if use_hybrid is not None
            else self.cfg.retrieval.hybrid_search.enabled
        )

        enable_rerank = (
            use_rerank
            if use_rerank is not None
            else self.cfg.retrieval.reranker.enabled
        )

        query = query.strip()

        logger.info(
            "Retrieval Pipeline | hybrid=%s | rerank=%s "
            "| candidate_k=%d | final_n=%d",
            enable_hybrid,
            enable_rerank,
            limit_k,
            limit_n,
        )

        # Buoc 1: Candidate Retrieval.
        active_retriever = (
            self.hybrid_retriever
            if enable_hybrid
            else self.retriever
        )

        filters = {
            key: value
            for key, value in (
                ("doc_id", doc_id),
                ("file_name", file_name),
                ("section_id", section_id),
            )
            if value is not None
        }
        candidate_chunks = active_retriever.retrieve(
            query=query,
            top_k=limit_k,
            **filters,
        )

        logger.info(
            "Candidate Retrieval tra ve %d chunks.",
            len(candidate_chunks),
        )

        if not candidate_chunks:
            logger.warning("Khong tim thay chunk nao.")
            return []

        if limit_n > len(candidate_chunks):
            logger.warning(
                "top_n=%d lon hon so candidates=%d; "
                "ket qua cuoi cung khong the vuot qua so candidates.",
                limit_n,
                len(candidate_chunks),
            )

        # Buoc 2: Reranking.
        if enable_rerank:
            if self.reranker is None:
                self.reranker = get_reranker(force=True)

            if self.reranker is not None:
                final_chunks = self.reranker.rerank(
                    query=query,
                    chunks=candidate_chunks,
                    top_n=limit_n,
                )
            else:
                logger.warning(
                    "Reranker không khả dụng; "
                    "Giữ các chunk từ retrieval ban đầu."
                )
                final_chunks = candidate_chunks[:limit_n]
        else:
            logger.info(
                "Tat Reranking; giu thu hang tu Retriever."
            )
            final_chunks = candidate_chunks[:limit_n]

        if self.context_expander is not None and final_chunks:
            final_chunks = self.context_expander.expand(query, final_chunks)
            logger.info(
                "Context expansion returned %d chunks within the configured prompt budget.",
                len(final_chunks),
            )

        logger.info(
            "Pipeline hoan tat: %d chunks cuoi cung.",
            len(final_chunks),
        )

        return final_chunks
