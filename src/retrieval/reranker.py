from typing import List, Optional
import torch
from sentence_transformers import CrossEncoder

from src.config import get_config
from src.logger import get_logger
from src.retrieval.base import BaseReranker
from src.schemas.response import RetrievedChunk

logger = get_logger("reranker")


class BGEReranker(BaseReranker):
    """Mô hình Cross-Encoder BAAI/bge-reranker-v2-m3 dùng để tái sắp xếp lại chunks."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        top_n: Optional[int] = None,
    ):
        cfg = get_config()
        self.model_name = model_name or cfg.retrieval.reranker.bge.model_name
        self.batch_size = batch_size or cfg.retrieval.reranker.bge.batch_size
        self.max_length = max_length or cfg.retrieval.reranker.bge.max_length
        self.top_n = top_n or cfg.retrieval.reranker.top_n

        target_device = device or cfg.retrieval.reranker.bge.device
        if target_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA không khả dụng cho Reranker. Tự động chuyển sang sử dụng CPU.")
            self.device = "cpu"
        else:
            self.device = target_device

        logger.info(f"Đang tải Reranker model: {self.model_name} trên thiết bị: {self.device}")
        try:
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
                trust_remote_code=True,
            )
            logger.info(f"Tải Reranker model {self.model_name} thành công!")
        except Exception as e:
            logger.error(f"Lỗi khi tải reranker model {self.model_name}: {e}")
            raise

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Tái sắp xếp danh sách chunks theo độ phù hợp ngữ nghĩa sâu sắc."""
        if not chunks:
            return []

        limit = top_n or self.top_n
        logger.info(f"Bắt đầu Reranking cho {len(chunks)} chunks với câu hỏi: '{query}' (top_n={limit})")

        # Chuẩn bị các cặp (câu hỏi, văn bản)
        pairs = [[query, chunk.text] for chunk in chunks]

        # Dự đoán điểm số tương quan
        scores = self.model.predict(
            sentences=pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Gắn điểm mới và sắp xếp giảm dần
        scored_pairs = list(zip(chunks, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        reranked_chunks: List[RetrievedChunk] = []
        for rank, (chunk, score) in enumerate(scored_pairs[:limit], start=1):
            # Tạo bản sao cập nhật điểm và thứ hạng mới
            new_chunk = chunk.model_copy(
                update={"score": float(score), "rank": rank}
            )
            reranked_chunks.append(new_chunk)

        logger.info(f"Hoàn tất Reranking. Lọc lại top {len(reranked_chunks)} chunks chất lượng nhất.")
        return reranked_chunks


class CohereReranker(BaseReranker):
    """Mô hình Cohere Rerank đa ngữ qua API (rerank-multilingual-v3.0)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        top_n: Optional[int] = None,
    ):
        cfg = get_config()
        self.api_key = api_key or cfg.retrieval.reranker.cohere.api_key
        self.model_name = model_name or cfg.retrieval.reranker.cohere.model_name
        self.top_n = top_n or cfg.retrieval.reranker.top_n

        if not self.api_key:
            raise ValueError(
                "Cohere API Key chưa được cấu hình. Vui lòng thêm COHERE_API_KEY vào file .env"
            )

        try:
            import cohere
            self.client = cohere.ClientV2(api_key=self.api_key)
        except ImportError:
            raise ImportError("Vui lòng cài đặt thư viện cohere: pip install cohere")

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []

        limit = top_n or self.top_n
        docs = [c.text for c in chunks]

        response = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=docs,
            top_n=limit,
        )

        reranked_chunks: List[RetrievedChunk] = []
        for rank, result in enumerate(response.results, start=1):
            original_chunk = chunks[result.index]
            new_chunk = original_chunk.model_copy(
                update={"score": float(result.relevance_score), "rank": rank}
            )
            reranked_chunks.append(new_chunk)

        return reranked_chunks


def get_reranker(force: bool = False) -> Optional[BaseReranker]:
    """Factory khởi tạo Reranker dựa theo cấu hình trong config.yaml."""
    cfg = get_config()
    if not cfg.retrieval.reranker.enabled and not force:
        return None

    provider = cfg.retrieval.reranker.provider.lower()
    if provider == "bge":
        return BGEReranker()
    elif provider == "cohere":
        return CohereReranker()
    else:
        raise ValueError(f"Provider Reranker '{provider}' chưa được hỗ trợ.")
