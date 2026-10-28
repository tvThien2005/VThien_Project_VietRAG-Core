"""
api_client.py - NƠI DUY NHẤT giao tiếp với FastAPI. app.py / ui.py không gọi requests trực tiếp.

 Yêu cầu backend thật        :  RAG_API_URL=http://localhost:8000

================================ HỢP ĐỒNG API (FastAPI) ================================
Mọi endpoint (trừ register/login) gửi kèm header:  Authorization: Bearer <access_token>
Lỗi: raise HTTPException(status_code, detail="...") -> UI hiển thị nguyên `detail`.

AUTH
  POST   /auth/register   {email, password, full_name}                 -> {message}
  POST   /auth/login      {email, password}                            -> {access_token, user:{email, full_name}}

SYSTEM (sidebar + header)
  GET    /system/status                                                -> {vector_db, total_vectors, embedding_model,
                                                                           collection, version}
DOCUMENTS (Kho Tri Thức)
  GET    /documents                                                    -> [{id, name, type:"pdf|docx|url", size:int|null,
                                                                            chunks:int, status:"ready|processing|failed",
                                                                            embedding_model, uploaded_at}]
  POST   /documents                multipart: file                     -> {id, ...như trên}
  DELETE /documents/{id}                                               -> 202 {id, status:"deleting", ...}
  GET    /documents/{id}/chunks                                        -> [{chunk_id, page, text}]
CHAT
  POST   /chat   {question, session_id|null, search_mode:"vector|bm25|hybrid", top_k, rerank, hyde}
                 -> {session_id, message_id,
                     answer,                      # markdown; chèn [1], [2]... để UI vẽ pill "[Nguồn 1: Score 0.91]"
                     note,                        # (tuỳ chọn) dòng "Lưu ý" in nghiêng cuối câu trả lời
                     sources:[{chunk_id, doc_name, page, score, text, used}],
                     rewritten_query,
                     metrics:{grounded, total_ms, retrieval_ms, contexts_used, contexts_total}}
HISTORY
  GET    /history?page=1&page_size=30                                  -> {items:[{session_id, title, updated_at}], has_more}
  GET    /history/{session_id}?limit=100 -> {messages:[{role, content, status, ts?, message_id?, sources?}], has_more}
  DELETE /history/{session_id}                                         -> {message}
FEEDBACK
  POST   /feedback   {message_id, rating:"up"|"down", comment, category}   -> {message}
CONFIG (Cấu hình Pipeline)
  Cấu hình là system-managed trong configs/config.yaml; UI chỉ hiển thị
  read-only và không gọi endpoint runtime để thay đổi pipeline.
METRICS (Giám sát)
  GET    /metrics -> {kpis:[{label, value, delta|null, tone, delta_tone, desc}],
                      latency:[{label, text, pct, color}]}   # tone: emerald|indigo|slate|amber
                                                             # color: purple|blue|amber|emerald
========================================================================================
"""
import os
from pathlib import Path

import requests
import yaml

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
TIMEOUT = 120

SPLITTERS = [
    "Recursive Character Text Splitter (Khuyên dùng)",
    "Semantic Chunking (Dựa trên nghĩa văn bản)",
    "Markdown / HTML Structure Aware",
    "Fixed Token Length Splitter",
]
EMBEDDINGS = [
    "OpenAI text-embedding-3-large (1536 dim)",
    "Cohere Embed v3 - Multilingual",
    "BGE-M3 (Dense + Sparse + Multi-Vector)",
]


def _load_pipeline_defaults() -> dict:
    """Đọc các giá trị retrieval từ cấu hình chuẩn, nhưng vẫn chạy được khi thiếu YAML."""
    defaults = {
        "splitter": SPLITTERS[0], "chunk_size": 512, "chunk_overlap": 64, "embedding": EMBEDDINGS[0],
        "search_mode": "vector", "top_k": 5, "top_n": 3, "rerank": False, "use_hybrid": False,
    }
    try:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
        with config_path.open("r", encoding="utf-8") as config_file:
            app_config = yaml.safe_load(config_file) or {}
        retrieval = app_config.get("retrieval", {})
        hybrid = retrieval.get("hybrid_search", {})
        reranker = retrieval.get("reranker", {})
        recursive = app_config.get("chunking", {}).get("recursive", {})
        chunk_strategy = app_config.get("chunking", {}).get("strategy", "recursive")
        splitter = SPLITTERS[1] if chunk_strategy == "semantic" else SPLITTERS[0]
        use_hybrid = bool(hybrid.get("enabled", defaults["use_hybrid"]))
        defaults.update(
            splitter=splitter,
            chunk_size=int(recursive.get("chunk_size", defaults["chunk_size"])),
            chunk_overlap=int(recursive.get("chunk_overlap", defaults["chunk_overlap"])),
            embedding=str(app_config.get("embedding", {}).get("model_name", defaults["embedding"])),
            top_k=int(retrieval.get("top_k", defaults["top_k"])),
            top_n=int(reranker.get("top_n", defaults["top_n"])),
            rerank=bool(reranker.get("enabled", defaults["rerank"])),
            use_hybrid=use_hybrid,
            search_mode="hybrid" if use_hybrid else "vector",
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        # Giữ default UI để app vẫn khởi động được nếu người dùng tách riêng thư mục app.
        pass
    return defaults


DEFAULT_CONFIG = _load_pipeline_defaults()


class APIError(Exception):
    pass


def _request(method: str, path: str, token: str | None = None, **kwargs):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=TIMEOUT, **kwargs)
    except requests.RequestException as e:
        raise APIError(f"Không kết nối được backend: {e}")
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise APIError(str(detail))
    return r.json() if r.content else {}


# =========================================================================== #
#  AUTH
# =========================================================================== #
def register(email: str, password: str, full_name: str) -> dict:
    return _request("POST", "/auth/register",
                    json={"email": email, "password": password, "full_name": full_name})


def login(email: str, password: str) -> dict:
    return _request("POST", "/auth/login", json={"email": email, "password": password})


# =========================================================================== #
#  SYSTEM
# =========================================================================== #
def get_status(token: str) -> dict:
    return _request("GET", "/system/status", token)


# =========================================================================== #
#  DOCUMENTS
# =========================================================================== #
def list_documents(token: str, page: int | None = None, page_size: int = 20, query: str = "") -> list[dict] | dict:
    if page is None:
        return _request("GET", "/documents", token)
    from urllib.parse import urlencode
    params = urlencode({"paginated": "true", "page": page, "page_size": page_size, "query": query})
    return _request("GET", f"/documents?{params}", token)


def upload_document(token: str, filename: str, data: bytes, mime: str) -> dict:
    return _request(
        "POST",
        "/documents",
        token,
        files={"file": (filename, data, mime)},
        data={"duplicate_policy": "reject"},
    )


def delete_document(token: str, doc_id: str) -> dict:
    return _request("DELETE", f"/documents/{doc_id}", token)


def get_chunks(token: str, doc_id: str, page: int | None = None, page_size: int = 25) -> list[dict] | dict:
    if page is None:
        return _request("GET", f"/documents/{doc_id}/chunks", token)
    return _request(
        "GET", f"/documents/{doc_id}/chunks?paginated=true&page={page}&page_size={page_size}", token
    )


# =========================================================================== #
#  CHAT
# =========================================================================== #
def ask(token: str, question: str, session_id: str | None, cfg: dict) -> dict:
    return _request("POST", "/chat", token, json={
        "question": question, "session_id": session_id, "search_mode": cfg["search_mode"],
        "top_k": cfg["top_k"], "rerank": cfg["rerank"], "hyde": False})


# =========================================================================== #
#  HISTORY
# =========================================================================== #
def list_sessions(token: str, page: int = 1, page_size: int = 30) -> list[dict]:
    result = _request("GET", f"/history?page={page}&page_size={page_size}", token)
    return result.get("items", [])


def get_session(token: str, session_id: str) -> dict:
    return _request("GET", f"/history/{session_id}", token)


def delete_session(token: str, session_id: str) -> dict:
    return _request("DELETE", f"/history/{session_id}", token)


# =========================================================================== #
#  FEEDBACK
# =========================================================================== #
def send_feedback(token: str, message_id: str, rating: str,
                  comment: str = "", category: str = "other") -> dict:
    payload = {"message_id": message_id, "rating": rating, "comment": comment, "category": category}
    return _request("POST", "/feedback", token, json=payload)


# =========================================================================== #
#  METRICS
# =========================================================================== #
def get_metrics(token: str) -> dict:
    return _request("GET", "/metrics", token)
