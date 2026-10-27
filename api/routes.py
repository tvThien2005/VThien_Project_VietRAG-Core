from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import PROJECT_ROOT, get_config
from src.ingestion.document_registry import DocumentRegistry
from src.ingestion.document_service import document_id_for_bytes, process_document_job
from src.ingestion.indexer import QdrantIndexer
from src.logger import get_logger
from src.monitoring.telemetry import QueryTelemetryStore
from src.chat.history_store import ChatHistoryStore
from api.auth import CurrentUser, UserStore, get_current_user, get_user_store, issue_access_token

router = APIRouter()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
logger = get_logger("api_routes")


@lru_cache(maxsize=1)
def get_document_registry() -> DocumentRegistry:
	return DocumentRegistry()


@lru_cache(maxsize=1)
def get_telemetry_store() -> QueryTelemetryStore:
	return QueryTelemetryStore()


@lru_cache(maxsize=1)
def get_chat_history() -> ChatHistoryStore:
	return ChatHistoryStore()


def _record_query_event(
	*,
	current_user: CurrentUser,
	cfg,
	status: str,
	search_mode: str,
	rerank: bool,
	top_k: int,
	top_n: int,
	source_count: int,
	timings_ms: dict | None = None,
	error_type: str | None = None,
) -> None:
	"""Telemetry must never prevent a chat response from being returned."""
	try:
		get_telemetry_store().record(
			owner_user_id=current_user.user_id,
			status=status,
			search_mode=search_mode,
			rerank_enabled=rerank,
			top_k=top_k,
			top_n=top_n,
			source_count=source_count,
			timings_ms=timings_ms,
			collection_name=getattr(getattr(cfg, "vector_store", None), "collection_name", None),
			llm_provider=getattr(getattr(cfg, "llm", None), "default_provider", None),
			error_type=error_type,
		)
	except Exception:
		logger.exception("Không thể ghi telemetry cho query; bỏ qua để không ảnh hưởng câu trả lời.")


def _mark_history_error_safe(history: ChatHistoryStore, *, current_user: CurrentUser, message_id: str, error: Exception) -> None:
	"""A secondary SQLite failure must not hide the original RAG failure."""
	try:
		history.mark_user_message_error(
			owner_user_id=current_user.user_id,
			message_id=message_id,
			error_message=f"RAG request failed: {type(error).__name__}",
		)
	except Exception:
		logger.exception("Không thể đánh dấu lỗi trong lịch sử chat.")


def _document_response(record: dict) -> dict:
	"""Map persistent registry fields to the current Streamlit document contract."""
	return {
		"id": record["doc_id"],
		"job_id": record["job_id"],
		"name": record["file_name"],
		"type": record["file_type"],
		"size": record["size"],
		"chunks": record["chunk_count"],
		"status": record["status"],
		"embedding_model": record.get("embedding_model") or "-",
		"uploaded_at": record["created_at"],
		"updated_at": record["updated_at"],
		"error": record.get("error_message"),
	}


class RegisterRequest(BaseModel):
	email: str = Field(min_length=3, max_length=320)
	password: str = Field(min_length=6, max_length=256)
	full_name: str = Field(min_length=1, max_length=200)


class LoginRequest(BaseModel):
	email: str = Field(min_length=3, max_length=320)
	password: str = Field(min_length=1, max_length=256)


def _user_payload(user: CurrentUser) -> dict:
	return {"id": user.user_id, "email": user.email, "full_name": user.full_name}


def _owner_filter(owner_user_id: str) -> models.Filter:
	return models.Filter(
		must=[
			models.FieldCondition(
				key="owner_user_id",
				match=models.MatchValue(value=owner_user_id),
			)
		]
	)


@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, user_store: UserStore = Depends(get_user_store)) -> dict:
	try:
		user = user_store.register(
			email=request.email,
			password=request.password,
			full_name=request.full_name,
		)
	except FileExistsError as exc:
		raise HTTPException(status_code=409, detail=str(exc)) from exc
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc)) from exc
	return {"message": "Đăng ký thành công.", "user": _user_payload(user)}


@router.post("/auth/login")
def login(request: LoginRequest, user_store: UserStore = Depends(get_user_store)) -> dict:
	user = user_store.authenticate(email=request.email, password=request.password)
	if user is None:
		raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng.")
	return {"access_token": issue_access_token(user), "token_type": "bearer", "user": _user_payload(user)}


@router.get("/system/status")
def system_status(current_user: CurrentUser = Depends(get_current_user)) -> dict:
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
			count_filter=_owner_filter(current_user.user_id),
			exact=False,
		).count
		points, _ = client.scroll(
			collection_name=cfg.vector_store.collection_name,
			scroll_filter=_owner_filter(current_user.user_id),
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


@router.post("/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
	background_tasks: BackgroundTasks,
	file: UploadFile = File(...),
	duplicate_policy: Literal["reject", "replace"] = Form("reject"),
	current_user: CurrentUser = Depends(get_current_user),
) -> dict:
	"""Store an upload and enqueue safe append into the configured collection."""
	cfg = get_config()
	filename = Path(file.filename or "upload").name
	extension = Path(filename).suffix.lower()
	if extension not in set(cfg.data.supported_extensions):
		raise HTTPException(status_code=422, detail="Chỉ hỗ trợ PDF, DOCX hoặc TXT.")
	content = await file.read()
	if not content:
		raise HTTPException(status_code=422, detail="File tải lên đang trống.")
	if len(content) > MAX_UPLOAD_BYTES:
		raise HTTPException(status_code=413, detail="Kích thước file vượt quá 10 MB.")

	content_sha256 = document_id_for_bytes(content)
	registry = get_document_registry()
	existing = registry.find_by_owner_content(current_user.user_id, content_sha256)
	if existing and duplicate_policy == "reject":
		raise HTTPException(status_code=409, detail="Tài liệu này đã được index trong kho của bạn.")
	doc_id = existing["doc_id"] if existing else uuid4().hex
	upload_dir = PROJECT_ROOT / "data" / "raw" / "uploads" / current_user.user_id
	upload_dir.mkdir(parents=True, exist_ok=True)
	source_path = upload_dir / f"{doc_id}{extension}"
	source_path.write_bytes(content)

	try:
		record = registry.create_job(
			doc_id=doc_id,
			job_id=uuid4().hex,
			file_name=filename,
			file_type=extension.lstrip("."),
			size=len(content),
			source_path=str(source_path),
		owner_user_id=current_user.user_id,
		content_sha256=content_sha256,
			replace_existing=duplicate_policy == "replace",
		)
	except FileExistsError as exc:
		raise HTTPException(status_code=409, detail=str(exc)) from exc
	except ValueError as exc:
		raise HTTPException(status_code=409, detail=str(exc)) from exc

	background_tasks.add_task(
		process_document_job,
		registry=registry,
		doc_id=doc_id,
		source_path=source_path,
		file_name=filename,
		owner_user_id=current_user.user_id,
		replace_existing=duplicate_policy == "replace",
	)
	return _document_response(record)


@router.get("/documents")
def list_documents(current_user: CurrentUser = Depends(get_current_user)) -> list[dict]:
	return [
		_document_response(record)
		for record in get_document_registry().list_documents(current_user.user_id)
	]


@router.get("/documents/{doc_id}/chunks")
def document_chunks(doc_id: str, current_user: CurrentUser = Depends(get_current_user)) -> list[dict]:
	record = get_document_registry().get_owned(doc_id, current_user.user_id)
	if not record:
		raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
	if record["status"] != "ready":
		raise HTTPException(status_code=409, detail="Tài liệu chưa sẵn sàng để xem chunks.")

	try:
		indexer = QdrantIndexer(collection_name=record["collection_name"])
		points = indexer._document_points(record["collection_name"], doc_id, current_user.user_id)
	except Exception as exc:
		raise HTTPException(status_code=503, detail=f"Không thể đọc chunks từ Qdrant: {exc}") from exc

	def item(point) -> dict:
		payload = point.payload or {}
		return {
			"chunk_id": payload.get("chunk_id") or str(point.id),
			"text": payload.get("text", ""),
			"page": payload.get("page_number") or payload.get("page_start"),
			"chunk_index": payload.get("chunk_index", 0),
		}

	return [item(point) for point in sorted(points, key=lambda point: (point.payload or {}).get("chunk_index", 0))]


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, current_user: CurrentUser = Depends(get_current_user)) -> dict:
	"""Remove only one ready document's Qdrant points; keep source artifacts for audit."""
	registry = get_document_registry()
	record = registry.get_owned(doc_id, current_user.user_id)
	if not record:
		raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
	if record["status"] in {"queued", "processing"}:
		raise HTTPException(status_code=409, detail="Không thể xóa tài liệu đang được xử lý.")
	try:
		indexer = QdrantIndexer(collection_name=record["collection_name"])
		points = indexer._document_points(record["collection_name"], doc_id, current_user.user_id)
		if points:
			indexer.client.delete(
				collection_name=record["collection_name"],
				points_selector=models.PointIdsList(points=[str(point.id) for point in points]),
				wait=True,
			)
		registry.remove(doc_id)
	except Exception as exc:
		raise HTTPException(status_code=503, detail=f"Không thể xóa points Qdrant: {exc}") from exc
	return {"message": "Đã xóa points của tài liệu khỏi collection; file nguồn được giữ lại để audit."}


class ChatRequest(BaseModel):
	question: str = Field(min_length=1, max_length=4000)
	session_id: str | None = Field(default=None, min_length=1, max_length=64)
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

	section_path = metadata.get("section_path") or metadata.get("breadcrumb") or ""
	if isinstance(section_path, (list, tuple)):
		section_path = " > ".join(str(item) for item in section_path if str(item).strip())

	return {
		"chunk_id": chunk.chunk_id,
		"doc_name": file_name,
		"section_path": str(section_path),
		"page": chunk.page_number or metadata.get("page_start"),
		"score": float(chunk.score),
		"text": chunk.text,
		"used": True,
	}


@router.post("/chat")
def chat(request: ChatRequest, current_user: CurrentUser = Depends(get_current_user)) -> dict:
	request_started = perf_counter()
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

	# A user question is durable before RAG execution.  If generation fails we
	# retain only this terminal error state, never a half-written assistant row.
	history = get_chat_history()
	try:
		session_id, user_message_id = history.create_user_message(
			owner_user_id=current_user.user_id,
			question=question,
			session_id=request.session_id,
		)
	except PermissionError as exc:
		raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện.") from exc
	except Exception as exc:
		logger.exception("Không thể lưu câu hỏi vào lịch sử chat.")
		raise HTTPException(status_code=503, detail="Không thể lưu lịch sử cuộc trò chuyện.") from exc

	try:
		response = get_rag_pipeline().query(
			query_text=question,
			top_k=top_k,
			top_n=top_n,
			use_rerank=rerank,
			use_hybrid=search_mode == "hybrid",
		owner_user_id=current_user.user_id,
		)
	except ConnectionError as exc:
		_mark_history_error_safe(history, current_user=current_user, message_id=user_message_id, error=exc)
		_record_query_event(
			current_user=current_user, cfg=cfg, status="error", search_mode=search_mode, rerank=rerank,
			top_k=top_k, top_n=top_n, source_count=0,
			timings_ms={"total_ms": (perf_counter() - request_started) * 1000}, error_type=type(exc).__name__,
		)
		raise HTTPException(status_code=503, detail=str(exc)) from exc
	except Exception as exc:
		_mark_history_error_safe(history, current_user=current_user, message_id=user_message_id, error=exc)
		_record_query_event(
			current_user=current_user, cfg=cfg, status="error", search_mode=search_mode, rerank=rerank,
			top_k=top_k, top_n=top_n, source_count=0,
			timings_ms={"total_ms": (perf_counter() - request_started) * 1000}, error_type=type(exc).__name__,
		)
		raise

	sources = [_source_payload(chunk) for chunk in response.sources]
	latency_ms = round((response.latency_seconds or 0) * 1000)
	timings_ms = dict((getattr(response, "metadata", {}) or {}).get("timings_ms", {}) or {})
	timings_ms.setdefault("total_ms", float(latency_ms))
	metrics_payload = {
		"grounded": bool(sources),
		"total_ms": latency_ms,
		"contexts_used": len(sources),
		"contexts_total": top_k,
		"configured_top_n": top_n,
	}
	try:
		message_id = history.complete_response(
			owner_user_id=current_user.user_id,
			session_id=session_id,
			user_message_id=user_message_id,
			answer=response.answer,
			sources=sources,
		)
	except Exception as exc:
		logger.exception("Không thể hoàn tất lịch sử chat.")
		try:
			history.mark_user_message_error(
				owner_user_id=current_user.user_id,
				message_id=user_message_id,
				error_message=f"History write failed: {type(exc).__name__}",
			)
		except Exception:
			logger.exception("Không thể đánh dấu lỗi lịch sử chat.")
		raise HTTPException(status_code=503, detail="Không thể lưu câu trả lời vào lịch sử cuộc trò chuyện.") from exc

	_record_query_event(
		current_user=current_user,
		cfg=cfg,
		status="success" if sources else "no_context",
		search_mode=search_mode,
		rerank=rerank,
		top_k=top_k,
		top_n=top_n,
		source_count=len(sources),
		timings_ms=timings_ms,
	)
	return {
		"session_id": session_id,
		"message_id": message_id,
		"answer": response.answer,
		"note": None,
		"sources": sources,
		"rewritten_query": question,
		"metrics": metrics_payload,
	}


class FeedbackRequest(BaseModel):
	message_id: str = Field(min_length=1, max_length=64)
	rating: Literal["up", "down"]
	comment: str = Field(default="", max_length=2000)
	category: str = Field(default="other", min_length=1, max_length=80)


@router.get("/history")
def list_history(
	page: int = Query(default=1, ge=1),
	page_size: int = Query(default=30, ge=1, le=100),
	current_user: CurrentUser = Depends(get_current_user),
) -> dict:
	"""Return one owner-scoped page; never load every user conversation."""
	return get_chat_history().list_sessions(
		owner_user_id=current_user.user_id, page=page, page_size=page_size
	)


@router.get("/history/{session_id}")
def get_history_session(
	session_id: str,
	before_created_at: str | None = Query(default=None, max_length=64),
	limit: int = Query(default=100, ge=1, le=100),
	current_user: CurrentUser = Depends(get_current_user),
) -> dict:
	try:
		return get_chat_history().get_session_messages(
			owner_user_id=current_user.user_id,
			session_id=session_id,
			before_created_at=before_created_at,
			limit=limit,
		)
	except KeyError as exc:
		raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện.") from exc


@router.delete("/history/{session_id}")
def delete_history_session(session_id: str, current_user: CurrentUser = Depends(get_current_user)) -> dict:
	if not get_chat_history().delete_session(owner_user_id=current_user.user_id, session_id=session_id):
		raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện.")
	return {"message": "Đã xóa cuộc trò chuyện cùng messages và feedback liên quan."}


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
def save_feedback(request: FeedbackRequest, current_user: CurrentUser = Depends(get_current_user)) -> dict:
	try:
		get_chat_history().save_feedback(
			owner_user_id=current_user.user_id,
			message_id=request.message_id,
			rating=request.rating,
			comment=request.comment.strip(),
			category=request.category.strip(),
		)
	except KeyError as exc:
		raise HTTPException(status_code=404, detail="Không tìm thấy câu trả lời để gửi phản hồi.") from exc
	return {"message": "Đã lưu phản hồi."}


@router.get("/metrics")
def metrics(current_user: CurrentUser = Depends(get_current_user)) -> dict:
	"""Return only the authenticated user's operational telemetry."""
	return get_telemetry_store().dashboard(current_user.user_id)
