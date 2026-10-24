"""
api_client.py - NƠI DUY NHẤT giao tiếp với FastAPI. app.py / ui.py không gọi requests trực tiếp.

Chạy demo không cần backend :  RAG_MOCK=1 (mặc định)
Nối backend thật            :  RAG_MOCK=0 RAG_API_URL=http://localhost:8000

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
  POST   /documents/crawl          {url}                               -> {id, ...như trên}
  DELETE /documents/{id}                                               -> {message}      (cũng dùng cho nút "Hủy")
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
  GET    /history                                                      -> [{session_id, title, updated_at}]
  GET    /history/{session_id}   -> {messages:[{role, content, ts?, message_id?, note?, sources?, rewritten_query?, metrics?}]}
  DELETE /history/{session_id}                                         -> {message}
FEEDBACK
  POST   /feedback   {message_id|null, rating:"up"|"down"|null, comment, category}   -> {message}
CONFIG (Cấu hình Pipeline)
  GET    /config                                                       -> {splitter, chunk_size, chunk_overlap, embedding,
                                                                           search_mode, top_k, rerank, hyde}
  PUT    /config   {…như trên…}                                        -> {…}
  POST   /config/reset                                                 -> {…}
METRICS (Giám sát)
  GET    /metrics -> {kpis:[{label, value, delta|null, tone, delta_tone, desc}],
                      latency:[{label, text, pct, color}]}   # tone: emerald|indigo|slate|amber
                                                             # color: purple|blue|amber|emerald
========================================================================================
"""
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
import yaml

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
MOCK_MODE = os.getenv("RAG_MOCK", "1") == "1"
CHAT_API_MODE = os.getenv("RAG_CHAT_API", "0") == "1"
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
            retrieval = (yaml.safe_load(config_file) or {}).get("retrieval", {})
        hybrid = retrieval.get("hybrid_search", {})
        reranker = retrieval.get("reranker", {})
        use_hybrid = bool(hybrid.get("enabled", defaults["use_hybrid"]))
        defaults.update(
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
#  MOCK STORE (chỉ dùng khi RAG_MOCK=1) - dữ liệu lấy theo file giao diện mẫu
# =========================================================================== #
DEFAULT_TEST_ACCOUNT = {
    "email": "email",
    "password": "123456",
    "full_name": "Văn Thiện",
}

_DB = {
    "users": {
        DEFAULT_TEST_ACCOUNT["email"]: {
            "password": DEFAULT_TEST_ACCOUNT["password"],
            "full_name": DEFAULT_TEST_ACCOUNT["full_name"],
        }
    },
    "docs": {},
    "sessions": {},
    "feedback": [],
    "config": {},
}


def _user(token: str) -> str:
    if token not in _DB["users"]:
        raise APIError("Phiên đăng nhập không hợp lệ")
    return token  # mock: token chính là email


def _seed_docs() -> dict:
    now = time.time()
    rows = [
        {"id": "d1", "name": "Quy_trinh_Bao_hanh_2026.pdf", "type": "pdf", "size": 2516582, "chunks": 142,
         "status": "ready", "uploaded_at": "20/10/2026", "_t": 0},
        {"id": "d2", "name": "HDSD_He_Thong_Internal_v3.docx", "type": "docx", "size": 5347737, "chunks": 380,
         "status": "ready", "uploaded_at": "18/10/2026", "_t": 0},
        {"id": "d3", "name": "https://docs.company.com/api-v2", "type": "url", "size": None, "chunks": 89,
         "status": "processing", "uploaded_at": "Vừa xong", "_t": now},
    ]
    for r in rows:
        r["embedding_model"] = "text-embed-3-large"
    return {r["id"]: r for r in rows}


def _docs(token: str) -> dict:
    email = _user(token)
    return _DB["docs"].setdefault(email, _seed_docs())


_SOURCES = [
    {"chunk_id": "chk_98231", "doc_name": "Quy_trinh_Bao_hanh_2026.pdf", "page": 4, "score": 0.912, "used": True,
     "text": "Thiết bị gặp sự cố phần cứng nằm trong thời hạn 12 tháng kể từ ngày kích hoạt sẽ được đổi mới 1-đổi-1 "
             "trong vòng 3 ngày làm việc kể từ khi Trung tâm Dịch vụ nhận được sản phẩm."},
    {"chunk_id": "chk_98232", "doc_name": "Quy_trinh_Bao_hanh_2026.pdf", "page": 5, "score": 0.864, "used": True,
     "text": "Đối với trường hợp thiết bị đã qua thời hạn đổi mới 1-đổi-1 nhưng còn hạn bảo hành, thời gian tiếp nhận "
             "sửa chữa phần cứng tối đa là 7 ngày làm việc không tính thứ 7 và chủ nhật."},
    {"chunk_id": "chk_41007", "doc_name": "Bang_gia_linh_kien.xlsx", "page": 1, "score": 0.651, "used": False,
     "text": "Bảng giá linh kiện thay thế dành cho dịch vụ sửa chữa ngoài bảo hành năm 2026 áp dụng cho dòng sản phẩm "
             "Enterprise..."},
]


def _now() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M")


# =========================================================================== #
#  AUTH
# =========================================================================== #
def register(email: str, password: str, full_name: str) -> dict:
    if MOCK_MODE:
        if email in _DB["users"]:
            raise APIError("Email đã tồn tại")
        _DB["users"][email] = {"password": password, "full_name": full_name}
        return {"message": "ok"}
    return _request("POST", "/auth/register",
                    json={"email": email, "password": password, "full_name": full_name})


def login(email: str, password: str) -> dict:
    if MOCK_MODE:
        u = _DB["users"].get(email)
        if not u or u["password"] != password:
            raise APIError("Sai email hoặc mật khẩu")
        return {"access_token": email, "user": {"email": email, "full_name": u["full_name"]}}
    return _request("POST", "/auth/login", json={"email": email, "password": password})


# =========================================================================== #
#  SYSTEM
# =========================================================================== #
def get_status(token: str) -> dict:
    if MOCK_MODE and not CHAT_API_MODE:
        _user(token)
        return {"vector_db": "Qdrant", "total_vectors": 148209, "embedding_model": "text-embed-3-large",
                "collection": "Vietnamese_docs", "version": "v2.4"}
    return _request("GET", "/system/status", token)


# =========================================================================== #
#  DOCUMENTS
# =========================================================================== #
def list_documents(token: str) -> list[dict]:
    if MOCK_MODE:
        docs = _docs(token)
        for d in docs.values():
            if d["status"] == "processing" and time.time() - d["_t"] > 8:
                d["status"] = "ready"  # mock: xử lý xong sau 8 giây
        return [{k: v for k, v in d.items() if not k.startswith("_")} for d in docs.values()]
    return _request("GET", "/documents", token)


def upload_document(token: str, filename: str, data: bytes, mime: str) -> dict:
    if MOCK_MODE:
        doc = {"id": uuid.uuid4().hex[:8], "name": filename, "type": filename.rsplit(".", 1)[-1].lower(),
               "size": len(data), "chunks": max(1, len(data) // 4000), "status": "processing",
               "embedding_model": "text-embed-3-large", "uploaded_at": datetime.now().strftime("%d/%m/%Y"),
               "_t": time.time()}
        _docs(token)[doc["id"]] = doc
        return doc
    return _request("POST", "/documents", token, files={"file": (filename, data, mime)})


def crawl_url(token: str, url: str) -> dict:
    if MOCK_MODE:
        doc = {"id": uuid.uuid4().hex[:8], "name": url, "type": "url", "size": None, "chunks": 0,
               "status": "processing", "embedding_model": "text-embed-3-large",
               "uploaded_at": "Vừa xong", "_t": time.time()}
        _docs(token)[doc["id"]] = doc
        return doc
    return _request("POST", "/documents/crawl", token, json={"url": url})


def delete_document(token: str, doc_id: str) -> dict:
    if MOCK_MODE:
        _docs(token).pop(doc_id, None)
        return {"message": "ok"}
    return _request("DELETE", f"/documents/{doc_id}", token)


def get_chunks(token: str, doc_id: str) -> list[dict]:
    if MOCK_MODE:
        _user(token)
        return [{"chunk_id": f"chk_{doc_id}_{i}", "page": i,
                 "text": s["text"]} for i, s in enumerate(_SOURCES[:2], start=1)]
    return _request("GET", f"/documents/{doc_id}/chunks", token)


# =========================================================================== #
#  CHAT
# =========================================================================== #
def ask(token: str, question: str, session_id: str | None, cfg: dict) -> dict:
    if MOCK_MODE and not CHAT_API_MODE:
        email = _user(token)
        top_k = int(cfg.get("top_k", DEFAULT_CONFIG["top_k"]))
        top_n = int(cfg.get("top_n", DEFAULT_CONFIG["top_n"]))
        # Mô phỏng cùng hợp đồng với backend: lấy candidate theo Top-K, sau đó giữ Top-N.
        final_count = max(1, min(top_k, top_n))
        sid = session_id or uuid.uuid4().hex[:8]
        sess = _DB["sessions"].setdefault(email, {}).setdefault(
            sid, {"title": question[:40], "updated_at": _now(), "messages": []})
        res = {
            "session_id": sid, "message_id": uuid.uuid4().hex[:8],
            "answer": ("Theo quy định trong tài liệu chính sách của doanh nghiệp, quy trình bảo hành lỗi phần cứng "
                       "được quy định rõ như sau:\n\n"
                       "- Thiết bị gặp sự cố phần cứng thuộc thời hạn 12 tháng kể từ ngày kích hoạt sẽ được áp dụng "
                       "chính sách **1-đổi-1 trong vòng 3 ngày làm việc** [1].\n"
                       "- Đối với các trường hợp quá thời hạn 1-đổi-1, thời gian sửa chữa tối đa không vượt quá "
                       "**7 ngày làm việc** [2]."),
            "note": "Yêu cầu sản phẩm còn nguyên tem bảo hành và không có dấu hiệu va đập cơ học nặng. "
                    "(Câu trả lời mẫu từ chế độ mock)",
            "sources": [dict(s) for s in _SOURCES[:final_count]],
            "rewritten_query": "Hardware warranty policy terms and replacement procedure duration limits",
            "metrics": {"grounded": True, "total_ms": 840, "retrieval_ms": 42,
                        "contexts_used": min(2, final_count), "contexts_total": top_k,
                        "configured_top_n": top_n},
        }
        ts = datetime.now().strftime("%I:%M %p")
        sess["messages"] += [{"role": "user", "content": question, "ts": ts},
                             {"role": "assistant", "content": res["answer"], "ts": ts,
                              **{k: res[k] for k in ("message_id", "note", "sources", "rewritten_query", "metrics")}}]
        sess["updated_at"] = _now()
        return res
    return _request("POST", "/chat", token, json={
        "question": question, "session_id": session_id, "search_mode": cfg["search_mode"],
        "top_k": cfg["top_k"], "rerank": cfg["rerank"], "hyde": False})


# =========================================================================== #
#  HISTORY
# =========================================================================== #
def list_sessions(token: str) -> list[dict]:
    if MOCK_MODE:
        s = _DB["sessions"].get(_user(token), {})
        items = [{"session_id": k, "title": v["title"], "updated_at": v["updated_at"]} for k, v in s.items()]
        return sorted(items, key=lambda x: x["updated_at"], reverse=True)
    return _request("GET", "/history", token)


def get_session(token: str, session_id: str) -> dict:
    if MOCK_MODE:
        s = _DB["sessions"].get(_user(token), {}).get(session_id)
        if not s:
            raise APIError("Không tìm thấy phiên chat")
        return {"messages": s["messages"]}
    return _request("GET", f"/history/{session_id}", token)


def delete_session(token: str, session_id: str) -> dict:
    if MOCK_MODE:
        _DB["sessions"].get(_user(token), {}).pop(session_id, None)
        return {"message": "ok"}
    return _request("DELETE", f"/history/{session_id}", token)


# =========================================================================== #
#  FEEDBACK
# =========================================================================== #
def send_feedback(token: str, message_id: str | None = None, rating: str | None = None,
                  comment: str = "", category: str = "other") -> dict:
    payload = {"message_id": message_id, "rating": rating, "comment": comment, "category": category}
    if MOCK_MODE:
        _user(token)
        _DB["feedback"].append(payload)
        return {"message": "ok"}
    return _request("POST", "/feedback", token, json=payload)


# =========================================================================== #
#  CONFIG
# =========================================================================== #
def get_config(token: str) -> dict:
    if MOCK_MODE:
        return dict(_DB["config"].get(_user(token), DEFAULT_CONFIG))
    return _request("GET", "/config", token)


def save_config(token: str, cfg: dict) -> dict:
    if MOCK_MODE:
        _DB["config"][_user(token)] = dict(cfg)
        return dict(cfg)
    return _request("PUT", "/config", token, json=cfg)


def reset_config(token: str) -> dict:
    if MOCK_MODE:
        _DB["config"].pop(_user(token), None)
        return dict(DEFAULT_CONFIG)
    return _request("POST", "/config/reset", token)


# =========================================================================== #
#  METRICS
# =========================================================================== #
def get_metrics(token: str) -> dict:
    if MOCK_MODE:
        _user(token)
        return {
            "kpis": [
                {"label": "Faithfulness Score", "value": "98.4%", "delta": "↑ +1.2%", "tone": "emerald",
                 "delta_tone": "emerald", "desc": "Tỷ lệ trả lời không ảo giác (Hallucination-free)"},
                {"label": "Context Precision", "value": "91.2%", "delta": "↑ +2.5%", "tone": "indigo",
                 "delta_tone": "indigo", "desc": "Độ chính xác của các đoạn Chunk được trích xuất"},
                {"label": "Avg Retrieval Latency", "value": "48ms", "delta": "↓ -5ms", "tone": "slate",
                 "delta_tone": "emerald", "desc": "Thời gian truy vấn Vector DB Qdrant"},
                {"label": "Cost per Query", "value": "$0.0024", "delta": None, "tone": "amber",
                 "delta_tone": "amber", "desc": "Embedding + LLM Token consumption"},
            ],
            "latency": [
                {"label": "1. Query Expansion & Rewriting (HyDE)", "text": "120ms (14%)", "pct": 14, "color": "purple"},
                {"label": "2. Vector DB Search (Dense + BM25)", "text": "42ms (5%)", "pct": 5, "color": "blue"},
                {"label": "3. Cohere Reranking", "text": "85ms (10%)", "pct": 10, "color": "amber"},
                {"label": "4. LLM Generation (GPT-4o) Time to First Token", "text": "593ms (71%)", "pct": 71,
                 "color": "emerald"},
            ],
        }
    return _request("GET", "/metrics", token)
