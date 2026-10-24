from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Đại diện cho một đoạn văn bản (chunk) được tìm thấy kèm điểm số độ tương đồng."""

    chunk_id: str
    doc_id: Optional[str] = None
    text: str
    score: float
    rank: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def file_name(self) -> Optional[str]:
        return self.metadata.get("file_name")

    @property
    def page_number(self) -> Optional[int]:
        return self.metadata.get("page_number")

    @property
    def section_title(self) -> Optional[str]:
        return self.metadata.get("section_title")


class RAGResponse(BaseModel):
    """Đại diện cho phản hồi hoàn chỉnh của hệ thống RAG."""

    query: str
    answer: str
    sources: List[RetrievedChunk] = Field(default_factory=list)
    latency_seconds: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
