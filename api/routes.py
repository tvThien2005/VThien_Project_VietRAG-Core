from functools import lru_cache
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from src.config import get_config

router = APIRouter()


@router.get("/system/status")
def system_status() -> dict:
	cfg = get_config()
	client = None
	try:
		client = QdrantClient(
			host=cfg.vector_store.host,
			port=cfg.vector_store.port,
			api_key=cfg.vector_store.api_key,
			timeout=cfg.vector_store.timeout,
		)
		total_vectors = client.count(
			collection_name=cfg.vector_store.collection_name,
			exact=True,
		).count
		points, _ = client.scroll(
			collection_name=cfg.vector_store.collection_name,
			limit=1,
			with_payload=["embedding_model"],
			with_vectors=False,
		)
	except Exception as exc:
		raise HTTPException(
			status_code=503,
			detail=f"Không thể đọc trạng thái collection Qdrant: {exc}",
		) from exc
	finally:
		if client is not None:
			client.close()

	embedding_model = cfg.embedding.model_name
	if points:
		embedding_model = (points[0].payload or {}).get("embedding_model") or embedding_model

	return {
		"vector_db": "Qdrant",
		"total_vectors": total_vectors,
		"embedding_model": embedding_model,
		"collection": cfg.vector_store.collection_name,
		"version": cfg.app.version,
	}


class ChatRequest(BaseModel):
	question: str = Field(min_length=1, max_length=4000)
	session_id: str | None = None
	# None = dùng giá trị trong configs/config.yaml. UI hiện chỉ cho override
	# search mode, top_k và rerank; top_n luôn thuộc cấu hình backend.
	search_mode: Literal["vector", "hybrid", "bm25"] | None = None
	top_k: int | None = Field(default=None, ge=1, le=50)
	rerank: bool | None = None
	hyde: bool = False


@lru_cache(maxsize=1)
def get_rag_pipeline():
	from src.pipeline.rag_pipeline import RAGPipeline

	return RAGPipeline()


def _source_payload(chunk) -> dict:
	metadata = chunk.metadata or {}
	file_name = metadata.get("file_name")
	if not file_name:
		source = str(metadata.get("source") or "")
		file_name = source.replace("\\", "/").rsplit("/", 1)[-1] or "Không rõ tài liệu"

	return {
		"chunk_id": chunk.chunk_id,
		"doc_name": file_name,
		"page": chunk.page_number or metadata.get("page_start"),
		"score": float(chunk.score),
		"text": chunk.text,
		"used": True,
	}


@router.post("/chat")
def chat(request: ChatRequest) -> dict:
	cfg = get_config()
	question = request.question.strip()
	if not question:
		raise HTTPException(status_code=422, detail="Câu hỏi không được để trống.")
	search_mode = request.search_mode or (
		"hybrid" if cfg.retrieval.hybrid_search.enabled else "vector"
	)
	top_k = request.top_k if request.top_k is not None else cfg.retrieval.top_k
	rerank = request.rerank if request.rerank is not None else cfg.retrieval.reranker.enabled
	# Chưa có điều khiển Top-N ở UI: luôn dùng giá trị đã được validate từ YAML.
	top_n = cfg.retrieval.reranker.top_n

	if search_mode == "bm25":
		raise HTTPException(
			status_code=422,
			detail="RAGPipeline hiện hỗ trợ tìm kiếm vector hoặc hybrid, chưa có BM25 độc lập.",
		)

	try:
		response = get_rag_pipeline().query(
			query_text=question,
			top_k=top_k,
			top_n=top_n,
			use_rerank=rerank,
			use_hybrid=search_mode == "hybrid",
		)
	except ConnectionError as exc:
		raise HTTPException(status_code=503, detail=str(exc)) from exc

	sources = [_source_payload(chunk) for chunk in response.sources]
	latency_ms = round((response.latency_seconds or 0) * 1000)
	return {
		"session_id": request.session_id or uuid4().hex[:8],
		"message_id": uuid4().hex[:8],
		"answer": response.answer,
		"note": None,
		"sources": sources,
		"rewritten_query": question,
		"metrics": {
			"grounded": bool(sources),
			"total_ms": latency_ms,
			"contexts_used": len(sources),
			"contexts_total": top_k,
			"configured_top_n": top_n,
		},
	}
