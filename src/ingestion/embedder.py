from typing import List, Optional, Union
import torch
from sentence_transformers import SentenceTransformer

from src.config import get_config
from src.logger import get_logger
from src.schemas.document import Chunk

logger = get_logger("embedder")


class BGEEmbedder:
    """Module tạo vector embeddings sử dụng mô hình BAAI/bge-m3 hoặc sentence-transformers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize_embeddings: Optional[bool] = None,
    ):
        cfg = get_config()
        self.model_name = model_name or cfg.embedding.model_name
        self.batch_size = batch_size or cfg.embedding.batch_size
        self.normalize_embeddings = (
            normalize_embeddings
            if normalize_embeddings is not None
            else cfg.embedding.normalize_embeddings
        )

        # Tự động phát hiện và kiểm tra thiết bị tính toán (CUDA / CPU)
        target_device = device or cfg.embedding.device
        if target_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA không khả dụng trên máy tính này. Tự động chuyển sang sử dụng CPU.")
            self.device = "cpu"
        else:
            self.device = target_device

        logger.info(f"Đang tải Embedding model: {self.model_name} trên thiết bị: {self.device}")
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            # Giới hạn context length theo cấu hình (BGE-M3 hỗ trợ tối đa 8192)
            if hasattr(self.model, "max_seq_length"):
                self.model.max_seq_length = cfg.embedding.max_length
            logger.info(f"Tải model {self.model_name} thành công. Kích thước vector: {self.dimension}")
        except Exception as e:
            logger.error(f"Lỗi khi tải embedding model {self.model_name}: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Kích thước chiều của vector embedding."""
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> List[List[float]]:
        """Tạo vector embeddings cho danh sách các chuỗi văn bản."""
        if not texts:
            return []

        bs = batch_size or self.batch_size
        embeddings = self.model.encode(
            sentences=texts,
            batch_size=bs,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Tạo embedding cho câu truy vấn (Query)."""
        if not query or not query.strip():
            raise ValueError("Câu truy vấn (query) không được rỗng.")

        embedding = self.model.encode(
            sentences=query.strip(),
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def embed_chunks(
        self,
        chunks: List[Chunk],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> List[List[float]]:
        """Tạo embeddings trực tiếp cho danh sách đối tượng Chunk."""
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(
            texts=texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

