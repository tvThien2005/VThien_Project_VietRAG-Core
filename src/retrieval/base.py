from abc import ABC, abstractmethod
from typing import List, Optional

from src.schemas.response import RetrievedChunk


class BaseRetriever(ABC):
    """Interface cơ sở cho các bộ tìm kiếm tài liệu (Vector Retriever, BM25, Hybrid)."""

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Truy xuất danh sách chunks liên quan dựa vào câu hỏi truy vấn."""
        pass


class BaseReranker(ABC):
    """Interface cơ sở cho các mô hình xếp hạng lại (Cross-Encoder, Cohere Rerank)."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Tái sắp xếp và lọc ra top_n chunks có điểm tương quan ngữ nghĩa cao nhất."""
        pass
