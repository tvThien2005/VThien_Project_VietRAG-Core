from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, Field

from src.utils.text import count_words


class Document(BaseModel):
    """Đại diện cho một tài liệu thô hoặc 1 trang tài liệu được load vào hệ thống."""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return count_words(self.text)

    def to_llama_document(self) -> Any:
        """Chuyển đổi sang LlamaIndex Document nếu cần."""
        try:
            from llama_index.core import Document as LlamaDocument
            meta = dict(self.metadata)
            meta["doc_id"] = self.doc_id
            return LlamaDocument(
                text=self.text,
                doc_id=self.doc_id,
                extra_info=meta,
            )
        except ImportError:
            raise ImportError("llama-index-core chưa được cài đặt.")


class Chunk(BaseModel):
    """Đại diện cho một đoạn văn bản (chunk) sau khi cắt nhỏ từ Document."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    text: str
    chunk_index: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return count_words(self.text)

    def to_llama_node(self) -> Any:
        """Chuyển đổi sang LlamaIndex TextNode để nạp vào Indexer / VectorStore."""
        try:
            from llama_index.core.schema import TextNode
            meta = dict(self.metadata)
            meta["chunk_id"] = self.chunk_id
            meta["doc_id"] = self.doc_id
            meta["chunk_index"] = self.chunk_index
            meta["char_count"] = self.char_count
            meta["word_count"] = self.word_count
            return TextNode(
                id_=self.chunk_id,
                text=self.text,
                metadata=meta,
            )
        except ImportError:
            raise ImportError("llama-index-core chưa được cài đặt.")
