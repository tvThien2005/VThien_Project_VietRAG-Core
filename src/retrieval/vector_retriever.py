from typing import List, Optional

from qdrant_client import QdrantClient

from src.config import get_config
from src.ingestion.embedder import BGEEmbedder
from src.logger import get_logger
from src.retrieval.qdrant_filters import build_payload_filter
from src.retrieval.base import BaseRetriever
from src.schemas.response import RetrievedChunk

logger = get_logger("vector_retriever")


class VectorRetriever(BaseRetriever):
    """Bộ truy xuất tài liệu sử dụng độ tương đồng vector trên Qdrant."""

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        dense_vector_name: Optional[str] = None,
    ):
        self.cfg = get_config()
        self.collection_name = collection_name or self.cfg.vector_store.collection_name
        self.top_k = top_k or self.cfg.retrieval.top_k
        self.dense_vector_name = (
            dense_vector_name
            if dense_vector_name is not None
            else self.cfg.vector_store.dense_vector_name
        )
        self._collection_compatibility_checked = False

        # Khởi tạo hoặc tái sử dụng Embedder
        self.embedder = embedder or BGEEmbedder()

        # Khởi tạo kết nối Qdrant
        if client is not None:
            self.client = client
        else:
            self.client = QdrantClient(
                host=self.cfg.vector_store.host,
                port=self.cfg.vector_store.port,
                api_key=self.cfg.vector_store.api_key,
                timeout=self.cfg.vector_store.timeout,
            )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        owner_user_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        file_name: Optional[str] = None,
        section_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Tạo embedding cho câu query và tìm kiếm các điểm gần nhất trong Qdrant."""
        if not query or not query.strip():
            logger.warning("Câu truy vấn rỗng.")
            return []

        limit = top_k or self.top_k
        logger.info(f"Bắt đầu tìm kiếm vector cho câu hỏi: '{query}' (top_k={limit})")

        # 1. Tạo query embedding
        self._verify_collection_compatibility()
        query_vector = self.embedder.embed_query(query)

        query_filter = build_payload_filter(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            file_name=file_name,
            section_id=section_id,
        )
        query_options = {}
        if self.dense_vector_name is not None:
            query_options["using"] = self.dense_vector_name
        if query_filter is not None:
            query_options["query_filter"] = query_filter

        # 2. Thực hiện query_points trên Qdrant
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                **query_options,
            )
        except Exception as e:
            err_msg = str(e)
            if "10061" in err_msg or "refused" in err_msg.lower() or "connection" in err_msg.lower():
                user_msg = (
                    f"Không thể kết nối tới Qdrant tại {self.cfg.vector_store.url}. "
                    "Vui lòng mở ứng dụng Docker Desktop và chạy lệnh: docker compose up -d"
                )
                logger.error(user_msg)
                raise ConnectionError(user_msg) from e
            logger.error(f"Lỗi khi truy vấn Qdrant collection '{self.collection_name}': {e}")
            raise

        retrieved_chunks: List[RetrievedChunk] = []
        for idx, point in enumerate(response.points):
            payload = dict(point.payload or {})
            text = payload.get("text", "")
            doc_id = payload.get("doc_id")

            chunk = RetrievedChunk(
                chunk_id=str(point.id),
                doc_id=doc_id,
                text=text,
                score=float(point.score),
                rank=idx + 1,
                metadata=payload,
            )
            retrieved_chunks.append(chunk)

        logger.info(f"Đã tìm thấy {len(retrieved_chunks)} chunks tương đồng từ Qdrant.")
        return retrieved_chunks

    def _verify_collection_compatibility(self) -> None:
        if self._collection_compatibility_checked or self.dense_vector_name is None:
            return

        collection = self.client.get_collection(collection_name=self.collection_name)
        vectors = collection.config.params.vectors
        if isinstance(vectors, dict):
            vector_config = vectors.get(self.dense_vector_name)
        else:
            vector_config = vectors
        if vector_config is None:
            raise ValueError(
                f"Collection '{self.collection_name}' does not define dense vector "
                f"'{self.dense_vector_name}'."
            )
        if vector_config.size != self.embedder.dimension:
            raise ValueError(
                f"Collection '{self.collection_name}' uses vector dimension "
                f"{vector_config.size}, but embedder '{self.embedder.model_name}' "
                f"produces {self.embedder.dimension}."
            )

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if points:
            payload = points[0].payload or {}
            indexed_model = payload.get("embedding_model")
            indexed_dimension = payload.get("embedding_dimension")
            if indexed_model and indexed_model != self.embedder.model_name:
                raise ValueError(
                    f"Collection '{self.collection_name}' was indexed with "
                    f"'{indexed_model}', not '{self.embedder.model_name}'."
                )
            if indexed_dimension and indexed_dimension != self.embedder.dimension:
                raise ValueError(
                    f"Collection '{self.collection_name}' payload declares embedding "
                    f"dimension {indexed_dimension}, not {self.embedder.dimension}."
                )
            if self.dense_vector_name == "dense" and not indexed_model:
                raise ValueError(
                    f"Collection '{self.collection_name}' is missing embedding version "
                    "metadata required for named-vector retrieval."
                )
        self._collection_compatibility_checked = True
