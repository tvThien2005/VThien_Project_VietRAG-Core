"""
app.py - Enterprise RAG Studio (Streamlit), tái tạo giao diện mẫu.
Chạy:  streamlit run app.py
Mọi lời gọi backend đi qua api_client.py.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

import api_client as api
import ui as ui
from api_client import APIError

st.set_page_config(page_title="Enterprise RAG Studio Dashboard", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

THEME = {"theme.base": "dark", "theme.primaryColor": "#4f46e5", "theme.backgroundColor": "#0b0f19",
         "theme.secondaryBackgroundColor": "#111827", "theme.textColor": "#e2e8f0"}

# JS chạy trong trang cha: tay nắm kéo đổi độ rộng sidebar (nhớ bằng localStorage, nhấp đúp để đặt lại 256px)
SIDEBAR_JS = """
(function(){
 var root=document.documentElement,MIN=224,MAX=480,DEF=256,drag=false;
 var saved=parseInt(localStorage.getItem('rag_sbw'),10);
 if(saved>=MIN&&saved<=MAX)root.style.setProperty('--sbw',saved+'px');
 var h=document.createElement('div');h.id='sb-handle';h.title='Kéo để đổi độ rộng (nhấp đúp để đặt lại)';
 document.body.appendChild(h);
 function setW(w){root.style.setProperty('--sbw',w+'px');}
 h.addEventListener('mousedown',function(e){drag=true;h.classList.add('on');
  document.body.style.userSelect='none';document.body.style.cursor='col-resize';e.preventDefault();});
 window.addEventListener('mousemove',function(e){if(drag)setW(Math.min(MAX,Math.max(MIN,e.clientX)));});
 window.addEventListener('mouseup',function(){if(!drag)return;drag=false;h.classList.remove('on');
  document.body.style.userSelect='';document.body.style.cursor='';
  localStorage.setItem('rag_sbw',parseInt(getComputedStyle(root).getPropertyValue('--sbw'),10));});
 h.addEventListener('dblclick',function(){setW(DEF);localStorage.setItem('rag_sbw',DEF);});
})();
"""

ALLOWED_EXT = ["pdf", "docx"]
MAX_MB = 10
NAV = [("chat", "Chat & Playground"), ("documents", "Kho Tri Thức (KB)"),
       ("config", "Cấu hình Pipeline"), ("metrics", "Giám sát & Metrics")]
TITLES = {"chat": "RAG Chat Playground", "documents": "Quản lý Tài liệu & Vector Database",
          "config": "Cấu hình RAG & Chunking Parameters", "metrics": "Đánh giá & Analytics Metrics"}
DOC_COLS = [3.4, 1.1, 1.4, 2.1, 1.9, 1.2, 1.9]
DOC_HEAD = ["Tên Tài Liệu", "Kích Thước", "Số Chunks", "Trạng Thái Vector", "Embedding Model", "Ngày Tải Lên",
            "Thao Tác"]

ss = st.session_state


def run_js(html_doc: str):
    """Chạy một đoạn <script> ẩn (st.iframe ở bản mới, components.html ở bản cũ)."""
    if hasattr(st, "iframe"):
        st.iframe(html_doc, height=1)
    else:
        components.html(html_doc, height=0)


def ensure_theme():
    """Nếu config.toml không được đọc (chạy lệnh từ thư mục khác), nạp theme tối rồi chạy lại 1 lần."""
    try:
        changed = False
        for k, v in THEME.items():
            if st.get_option(k) != v:
                st._config.set_option(k, v)
                changed = True
        if changed and not ss.get("_themed"):
            ss["_themed"] = True
            st.rerun()
    except Exception:
        pass


ensure_theme()
for _k, _v in {"sb_open": True, "token": None, "user": None, "tab": "chat", "drawer_open": True, "sel_msg": None,
               "messages": [], "session_id": None, "fb": {}, "dz_nonce": 0, "scroll": False,
               "cfg": dict(api.DEFAULT_CONFIG), "chat_task": None, "auth_task": None,
               "auth_notice": None, "document_count": 0, "chat_upload_nonce": 0,
               "chat_upload_job": None, "show_logout_dialog": False, "delete_doc_to_confirm": None}.items():
    ss.setdefault(_k, _v)

if "use_hybrid" not in ss.cfg:
    ss.cfg["use_hybrid"] = False
    if ss.cfg.get("search_mode") == "hybrid":
        ss.cfg["search_mode"] = "vector"


# =========================================================================== #
# Callbacks
# =========================================================================== #
def set_tab(t):
    ss.tab = t


def toggle_sidebar():
    ss.sb_open = not ss.sb_open


def close_drawer():
    ss.drawer_open = False


def open_drawer():
    """Reopen the retrieved-chunks panel without changing the selected answer."""
    ss.drawer_open = True


def toggle_ctx(i):
    if ss.drawer_open and ss.sel_msg == i:
        ss.drawer_open = False
    else:
        ss.drawer_open, ss.sel_msg = True, i


def new_chat():
    ss.messages, ss.session_id, ss.sel_msg, ss.hist_sel = [], None, None, None


def _last_ai(msgs):
    idx = [k for k, m in enumerate(msgs) if m["role"] == "assistant"]
    return idx[-1] if idx else None


def load_session():
    sid = ss.get("hist_sel")
    if not sid:
        return
    try:
        ss.messages = api.get_session(ss.token, sid)["messages"]
        ss.session_id, ss.sel_msg = sid, _last_ai(ss.messages)
    except APIError as e:
        st.toast(str(e), icon="⚠️")


def delete_session():
    try:
        api.delete_session(ss.token, ss.session_id)
        new_chat()
    except APIError as e:
        st.toast(str(e), icon="⚠️")


def send_fb(mid, rating):
    try:
        api.send_feedback(ss.token, message_id=mid, rating=rating)
        ss.fb[mid] = rating
        st.toast("Cảm ơn phản hồi của bạn!", icon="✅")
    except APIError as e:
        st.toast(str(e), icon="⚠️")


def logout():
    task = ss.get("chat_task")
    if task:
        task["future"].cancel()
    for k in ("token", "user", "session_id", "sel_msg"):
        ss[k] = None
    ss.messages, ss.fb, ss.tab, ss.chat_task, ss.document_count = [], {}, "chat", None, 0


def request_logout():
    ss.show_logout_dialog = True


def dismiss_logout_dialog():
    ss.show_logout_dialog = False


def request_delete_document(doc: dict):
    ss.delete_doc_to_confirm = {"id": doc["id"], "name": doc["name"], "status": doc["status"]}


def dismiss_delete_dialog():
    ss.delete_doc_to_confirm = None


def clear_chat_upload_job():
    ss.chat_upload_job = None


def set_hybrid_search():
    """Apply the chat-only Hybrid Search toggle to this session's request options."""
    enabled = bool(ss.get("dk_hybrid", False))
    ss.cfg["use_hybrid"] = enabled
    ss.cfg["search_mode"] = "hybrid" if enabled else "vector"


# =========================================================================== #
# Dialogs
# =========================================================================== #
def ingest(files, on_queued=None) -> int:
    ok = 0
    for f in files:
        ext = f.name.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXT:
            st.toast(f"'{f.name}': chỉ hỗ trợ PDF, DOCX", icon="⚠️")
        elif f.size > MAX_MB * 1024 * 1024:
            st.toast(f"'{f.name}' vượt quá {MAX_MB}MB", icon="⚠️")
        else:
            try:
                queued_document = api.upload_document(ss.token, f.name, f.getvalue(), f.type)
                if on_queued is not None:
                    on_queued(queued_document)
                ok += 1
            except APIError as e:
                st.toast(f"{f.name}: {e}", icon="⚠️")
    if ok:
        # Danh sách có thể đang được cache trong tab Kho tri thức.
        # Xóa cache sau khi queue job để lần render kế tiếp thấy trạng thái mới.
        cached_documents.clear()
    return ok


@st.dialog("Xác nhận đăng xuất", on_dismiss=dismiss_logout_dialog)
def logout_confirm_dialog():
    st.warning("Bạn sẽ cần đăng nhập lại để tiếp tục sử dụng hệ thống.")
    cancel, confirm = st.columns(2)
    if cancel.button("Hủy", key="logout_cancel"):
        dismiss_logout_dialog()
        st.rerun()
    if confirm.button("Đăng xuất", key="logout_confirm_button", type="primary"):
        dismiss_logout_dialog()
        logout()
        st.rerun()


@st.dialog("Xác nhận xóa tài liệu", on_dismiss=dismiss_delete_dialog)
def delete_document_confirm_dialog(doc: dict):
    st.warning(f"Xóa '{doc['name']}'?")
    st.caption("Toàn bộ chunks/vectors của tài liệu này sẽ bị xóa khỏi collection hiện tại.")
    cancel, confirm = st.columns(2)
    if cancel.button("Hủy", key="delete_doc_cancel"):
        dismiss_delete_dialog()
        st.rerun()
    if confirm.button("Xóa tài liệu", key="delete_doc_confirm", type="primary"):
        try:
            api.delete_document(ss.token, doc["id"])
            cached_documents.clear()
            dismiss_delete_dialog()
            st.toast("Đã xóa tài liệu và các chunks liên quan.", icon="✅")
            st.rerun()
        except APIError as e:
            st.error(str(e))


@st.dialog("Tải lên Tài liệu")
def upload_dialog():
    files = st.file_uploader(f"PDF hoặc DOCX, tối đa {MAX_MB}MB/file", type=ALLOWED_EXT,
                             accept_multiple_files=True, key="dlg_up")
    if files and st.button("Tải lên", key="pri_dlg_up"):
        if ingest(files):
            st.rerun()


@st.dialog("Kết nối Web Crawler")
def crawler_dialog():
    url = st.text_input("URL cần thu thập", placeholder="https://docs.company.com/api-v2", key="crawl_url")
    if st.button("Bắt đầu thu thập", key="pri_crawl_go"):
        if not url.startswith(("http://", "https://")):
            st.warning("URL phải bắt đầu bằng http:// hoặc https://")
            return
        try:
            api.crawl_url(ss.token, url)
            st.rerun()
        except APIError as e:
            st.error(str(e))


@st.dialog("Chunks của tài liệu", width="large")
def chunks_dialog(doc):
    st.markdown(f"**{doc['name']}**")
    try:
        chunks = api.get_chunks(ss.token, doc["id"])
    except APIError as e:
        st.error(str(e))
        return
    for i, c in enumerate(chunks, 1):
        st.markdown(ui.chunk_card(i, {"text": c["text"], "doc_name": doc["name"], "page": c.get("page"),
                                      "score": 1.0, "used": True}), unsafe_allow_html=True)


# =========================================================================== #
# Auth page
# =========================================================================== #
def _login_request(email, pwd):
    """Worker an toàn: chỉ thực hiện I/O, không truy cập st.session_state."""
    res = api.login(email, pwd)
    # Pipeline configuration is system-managed in configs/config.yaml; it is
    # intentionally not a per-user backend resource.
    return res, dict(api.DEFAULT_CONFIG)


def do_login(email, pwd):
    if ss.get("auth_task"):
        return
    ss.auth_task = {
        "kind": "login",
        "future": api_request_executor().submit(_login_request, email, pwd),
    }


def register_account(email, pwd, name):
    if ss.get("auth_task"):
        return
    ss.auth_task = {
        "kind": "register",
        "future": api_request_executor().submit(api.register, email, pwd, name),
    }


def auth_progress():
    """Polling nhỏ cho xác thực; không để một request HTTP làm UI stale."""
    refresh_interval = 0.5 if ss.get("auth_task") else None

    @st.fragment(run_every=refresh_interval)
    def _render_auth_progress():
        task = ss.get("auth_task")
        if not task:
            notice = ss.pop("auth_notice", None)
            if notice:
                st.success(notice)
            return
        if not task["future"].done():
            st.info("Đang xử lý yêu cầu xác thực…")
            return
        try:
            result = task["future"].result()
        except APIError as e:
            ss.auth_task = None
            st.toast(str(e), icon="⚠️")
            st.rerun()
            return
        except Exception:
            ss.auth_task = None
            st.toast("Không thể kết nối dịch vụ xác thực. Vui lòng thử lại.", icon="⚠️")
            st.rerun()
            return

        if task["kind"] == "login":
            res, config = result
            ss.token, ss.user = res["access_token"], res["user"]
            ss.cfg = {**api.DEFAULT_CONFIG, **config}
        else:
            ss.auth_notice = "Tạo tài khoản thành công! Hãy chuyển sang tab Đăng nhập."
        ss.auth_task = None
        st.rerun()

    _render_auth_progress()


def auth_page():
    pending = bool(ss.get("auth_task"))
    with st.container(key="auth"):
        st.markdown(ui.logo_html(), unsafe_allow_html=True)
        if api.MOCK_MODE and not api.AUTH_API_MODE:
            st.markdown(ui.auth_note_html(), unsafe_allow_html=True)
        t_in, t_up = st.tabs(["Đăng nhập", "Tạo tài khoản"])
        with t_in:
            with st.form("login_form"):
                email = st.text_input("Email", key="lg_email", disabled=pending)
                pwd = st.text_input("Mật khẩu", type="password", key="lg_pwd", disabled=pending)
                ok = st.form_submit_button("Đang xử lý..." if pending else "Đăng nhập", disabled=pending)
            if ok:
                do_login(email, pwd)
                st.rerun()
        with t_up:
            with st.form("register_form"):
                name = st.text_input("Họ tên", key="rg_name", disabled=pending)
                email = st.text_input("Email", key="rg_email", disabled=pending)
                pwd = st.text_input("Mật khẩu", type="password", key="rg_pwd", disabled=pending)
                pwd2 = st.text_input("Nhập lại mật khẩu", type="password", key="rg_pwd2", disabled=pending)
                ok = st.form_submit_button("Đang xử lý..." if pending else "Tạo tài khoản", disabled=pending)
            if ok:
                if not (name and email and pwd):
                    st.warning("Vui lòng điền đầy đủ thông tin.")
                elif len(pwd) < 6:
                    st.warning("Mật khẩu tối thiểu 6 ký tự.")
                elif pwd != pwd2:
                    st.warning("Mật khẩu nhập lại không khớp.")
                else:
                    register_account(email, pwd, name)
                    st.rerun()
        auth_progress()


# =========================================================================== #
# Sidebar
# =========================================================================== #
def sidebar(status):
    if not ss.sb_open:  # đang thu gọn: chỉ hiện nút mở lại ở góc trái header
        st.button("Mở thanh bên", key="ico_sb_open", on_click=toggle_sidebar, help="Mở thanh bên")
        return
    with st.container(key="sb"):
        st.button("Thu gọn", key="ico_sb_close", on_click=toggle_sidebar, help="Thu gọn thanh bên")
        with st.container(key="side_top"):
            st.markdown(ui.logo_html(status.get("version", "v2.4")), unsafe_allow_html=True)
            with st.container(key="nav"):
                for key, label in NAV:
                    st.button(label, key=f"nav_{key}", on_click=set_tab, args=(key,))
        with st.container(key="side_bot"):
            st.markdown(ui.status_html(status), unsafe_allow_html=True)
            with st.container(key="side_user"):
                c1, c2 = st.columns([5, 1])
                c1.markdown(ui.user_html(ss.user), unsafe_allow_html=True)
                c2.button("Đăng xuất", key="ico_logout", on_click=request_logout, help="Đăng xuất")


# =========================================================================== #
# Tab 1: Chat
# =========================================================================== #
def history_bar():
    try:
        sessions = api.list_sessions(ss.token)
    except APIError:
        sessions = []
    titles = {s["session_id"]: f'{s["title"] or "Cuộc trò chuyện"} · {s["updated_at"]}' for s in sessions}
    with st.container(key="histbar"):
        c1, c2, c3 = st.columns([1.7, 3, 0.5])
        c1.button("Cuộc trò chuyện mới", key="hist_new", on_click=new_chat)
        c2.selectbox("Lịch sử", list(titles), index=None, format_func=lambda k: titles.get(k, k),
                     placeholder="Lịch sử trò chuyện", key="hist_sel", on_change=load_session,
                     label_visibility="collapsed")
        c3.button("Xóa", key="hist_del", on_click=delete_session, disabled=not ss.session_id,
                  help="Xóa cuộc trò chuyện hiện tại")


def message_actions(i, m):
    with st.container(key=f"rowr_{i}"):
        mid = m.get("message_id")
        if mid:
            st.button("Hữu ích", key=f"fbup_{i}", on_click=send_fb, args=(mid, "up"), help="Câu trả lời hữu ích")
            st.button("Chưa tốt", key=f"fbdn_{i}", on_click=send_fb, args=(mid, "down"), help="Câu trả lời chưa tốt")
        st.button("Xem Context Trích Dẫn", key=f"ctx_{i}", on_click=toggle_ctx, args=(i,))


def drawer_msg():
    msgs, i = ss.messages, ss.sel_msg
    if i is None or i >= len(msgs) or msgs[i]["role"] != "assistant":
        i = _last_ai(msgs)
    return msgs[i] if i is not None else None


@st.cache_resource
def api_request_executor() -> ThreadPoolExecutor:
    """Executor dùng chung cho UI requests, không truy cập Streamlit state từ worker."""
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-chat")


def send_question(q):
    """Khởi chạy request RAG ở nền để một Streamlit rerun không khóa toàn bộ UI."""
    if ss.get("chat_task"):
        return
    umsg = {"role": "user", "content": q, "ts": datetime.now().strftime("%I:%M %p")}
    # Chỉ truyền dữ liệu bất biến vào worker; session_state chỉ được sửa ở script thread.
    future = api_request_executor().submit(api.ask, ss.token, q, ss.session_id, dict(ss.cfg))
    ss.chat_task = {"future": future, "user_message": umsg}
    st.rerun()


def dock():
    cfg = ss.cfg
    pending = bool(ss.get("chat_task"))
    with st.container(key="dock"):
        with st.container(key="dock_in"):
            with st.container(key="dockrow"):
                c1, c2, c3 = st.columns([0.27, 0.2, 0.53])
                cfg["use_hybrid"] = c1.checkbox("Hybrid Search", value=cfg["use_hybrid"], key="dk_hybrid",
                                                on_change=set_hybrid_search)
                cfg["search_mode"] = "hybrid" if cfg["use_hybrid"] else (
                    "vector" if cfg["search_mode"] == "hybrid" else cfg["search_mode"])
                cfg["rerank"] = c2.checkbox("Reranker", value=cfg["rerank"], key="dk_rerank")
                c3.markdown(f'<div class="topk">Top-K: <strong>{cfg["top_k"]}</strong> Chunks</div>',
                            unsafe_allow_html=True)
            with st.container(key="formbox"):
                with st.form("chat_form", clear_on_submit=True):
                    q = st.text_area("Câu hỏi", key="chat_q", height=68, label_visibility="collapsed",
                                     placeholder="Hỏi bất kỳ điều gì dựa trên kho tài liệu đã upload...",
                                     disabled=pending)
                    with st.container(key="send_box"):
                        sent = st.form_submit_button("Đang xử lý..." if pending else "Gửi", disabled=pending)
                # The transparent uploader overlays the paperclip drawn by CSS.
                # Selecting a file queues the same ingestion flow as the KB page.
                with st.container(key="chat_file_upload"):
                    chat_file = st.file_uploader(
                        "Đính kèm tài liệu PDF hoặc DOCX",
                        type=ALLOWED_EXT,
                        key=f"chat_up_{ss.chat_upload_nonce}",
                        label_visibility="collapsed",
                        disabled=pending,
                    )
    if chat_file:
        filename = chat_file.name
        queued_records = []
        with st.spinner(f"Đang gửi '{filename}' để xử lý..."):
            queued = ingest([chat_file], on_queued=queued_records.append)
        # Reset the file widget whether the upload was accepted or rejected;
        # otherwise Streamlit would re-submit it on every rerun.
        ss.chat_upload_nonce += 1
        if queued and queued_records:
            record = queued_records[0]
            ss.chat_upload_job = {
                "id": record.get("id"),
                "job_id": record.get("job_id"),
                "name": record.get("name") or filename,
                "status": record.get("status", "queued"),
            }
        st.rerun()
    if sent and q.strip():
        send_question(q.strip())


def chat_feed():
    """Render chat và chỉ tự làm mới vùng này khi có một request đang chờ."""
    refresh_interval = 0.5 if ss.get("chat_task") else None

    @st.fragment(run_every=refresh_interval)
    def _render_chat_feed():
        task = ss.get("chat_task")
        if task and task["future"].done():
            try:
                res = task["future"].result()
            except APIError as e:
                ss.chat_task = None
                st.toast(str(e), icon="⚠️")
                st.rerun()
                return
            except Exception:
                ss.chat_task = None
                st.toast("Không thể nhận phản hồi từ dịch vụ RAG. Vui lòng thử lại.", icon="⚠️")
                st.rerun()
                return
            umsg = task["user_message"]
            ss.session_id = res["session_id"]
            ss.messages += [umsg, {"role": "assistant", "content": res["answer"], "ts": umsg["ts"],
                                   **{k: res.get(k) for k in ("message_id", "note", "sources", "rewritten_query", "metrics")}}]
            ss.sel_msg = len(ss.messages) - 1
            ss.scroll, ss.chat_task = True, None
            # Rerun nhanh một lần để dừng polling và cập nhật drawer/history; request RAG đã hoàn tất.
            st.rerun()

        for i, m in enumerate(ss.messages):
            if m["role"] == "user":
                st.markdown(ui.user_bubble(m), unsafe_allow_html=True)
            else:
                st.markdown(ui.ai_message(m), unsafe_allow_html=True)
                message_actions(i, m)
        if task:
            st.markdown(ui.user_bubble(task["user_message"]) + ui.thinking_html(), unsafe_allow_html=True)

    _render_chat_feed()


@st.fragment(run_every=3)
def chat_upload_status():
    """Show a persistent, polling upload state instead of a fleeting toast."""
    job = ss.get("chat_upload_job")
    if not job:
        return
    current = job
    try:
        documents = cached_documents(ss.token)
        current = next((doc for doc in documents if doc.get("id") == job.get("id")), job)
    except APIError:
        pass

    state = current.get("status", "queued")
    filename = current.get("name") or job.get("name") or "tài liệu"
    if state in {"queued", "processing"}:
        with st.status(f"Đang xử lý '{filename}'", state="running", expanded=True):
            st.write("Tài liệu đã được gửi. Hệ thống đang parse, chunk và embedding vào kho tri thức.")
            st.caption("Bạn có thể tiếp tục chat; tài liệu sẽ được dùng khi trạng thái chuyển sang sẵn sàng.")
    elif state == "ready":
        with st.status(f"'{filename}' đã sẵn sàng", state="complete", expanded=True):
            st.write("Chunking và embedding đã hoàn tất. Các câu hỏi mới có thể truy xuất tài liệu này.")
            st.button("Ẩn thông báo", key="chat_upload_status_close", on_click=clear_chat_upload_job)
    else:
        with st.status(f"Không thể xử lý '{filename}'", state="error", expanded=True):
            st.write(current.get("error") or "Job ingest thất bại. Hãy kiểm tra lại tài liệu rồi thử lại.")
            st.button("Ẩn thông báo", key="chat_upload_status_close", on_click=clear_chat_upload_job)


def page_chat():
    with st.container(key="chatpane"):
        history_bar()
        chat_upload_status()
        st.markdown(ui.banner_html(), unsafe_allow_html=True)
        chat_feed()
    dock()
    if ss.drawer_open:
        st.markdown(ui.drawer_html(drawer_msg(), ss.cfg["use_hybrid"]), unsafe_allow_html=True)
        st.button("Đóng", key="drawer_close", on_click=close_drawer)
    else:
        # Always leave an obvious way to reopen the right panel after it has
        # been closed.  It is disabled only until an answer has contexts.
        st.button("Mở chunks", key="drawer_reopen", on_click=open_drawer,
                  disabled=not bool((drawer_msg() or {}).get("sources")),
                  help="Hiển thị lại các chunk đã được truy xuất")
    if ss.pop("scroll", False):
        run_js("<script>const d=window.parent.document;const m=d.querySelector('[data-testid=\"stMain\"]')"
               "||d.querySelector('section.main');if(m)m.scrollTo({top:m.scrollHeight,behavior:'smooth'});</script>")


# =========================================================================== #
# Tab 2: Kho tri thức
# =========================================================================== #
@st.fragment(run_every=3)
def page_docs():
    try:
        docs = cached_documents(ss.token)
        ss.document_count = len(docs)
    except APIError as e:
        st.error(str(e))
        return

    with st.container(key="pg_docs"):
        c1, c2 = st.columns([7, 3])
        c1.markdown(ui.page_head("Kho Tri Thức Doanh Nghiệp",
                                 "Quản lý tài liệu nguồn, đồng bộ dữ liệu và quy trình Chunking / Embedding"),
                    unsafe_allow_html=True)
        with c2:
            with st.container(key="rowf_docs"):
                if st.button("Tải lên Tài liệu", key="pri_upload"):
                    upload_dialog()

        with st.container(key="dz"):
            st.markdown(ui.dropzone_html(MAX_MB), unsafe_allow_html=True)
            files = st.file_uploader("Tải lên", type=ALLOWED_EXT, accept_multiple_files=True,
                                     key=f"dz_up_{ss.dz_nonce}", label_visibility="collapsed")
        if files:
            n = ingest(files)
            ss.dz_nonce += 1
            if n:
                st.toast(f"Đã tải lên {n} tài liệu", icon="✅")
            st.rerun()

        with st.container(key="tbl"):
            with st.container(key="tbl_bar"):
                b1, b2 = st.columns([1, 1])
                kw = b1.text_input("Tìm", key="tbl_q", placeholder="Tìm kiếm tài liệu...",
                                   label_visibility="collapsed")
                b2.markdown(f'<div class="cnt">Đã chọn {len(docs)} tài liệu</div>', unsafe_allow_html=True)
            with st.container(key="tbl_head"):
                for col, h in zip(st.columns(DOC_COLS), DOC_HEAD):
                    col.markdown(ui.th(h, right=(h == "Thao Tác")), unsafe_allow_html=True)
            shown = [d for d in docs if kw.lower() in d["name"].lower()]
            if not shown:
                st.markdown('<div class="td" style="text-align:center;color:var(--s500)">Chưa có tài liệu nào.</div>',
                            unsafe_allow_html=True)
            for d in shown:
                doc_row(d)
def doc_row(d):
    k = d["id"]
    with st.container(key=f"trow_{k}"):
        c = st.columns(DOC_COLS)
        c[0].markdown(ui.td_name(d), unsafe_allow_html=True)
        c[1].markdown(ui.td(ui.fmt_size(d.get("size")), style="color:var(--s400)"), unsafe_allow_html=True)
        c[2].markdown(ui.td(f'{d.get("chunks", 0)} Chunks', "mono"), unsafe_allow_html=True)
        c[3].markdown(ui.td_status(d["status"]), unsafe_allow_html=True)
        c[4].markdown(ui.td(d.get("embedding_model", ""), "mono", "color:var(--s400)"), unsafe_allow_html=True)
        c[5].markdown(ui.td(d.get("uploaded_at", ""), style="color:var(--s400)"), unsafe_allow_html=True)
        with c[6]:
            with st.container(key=f"rowf_a_{k}"):
                if st.button("Xem Chunks", key=f"lnk_view_{k}", disabled=d["status"] != "ready"):
                    chunks_dialog(d)
                if st.button("Hủy" if d["status"] == "processing" else "Xóa", key=f"dng_del_{k}"):
                    request_delete_document(d)
                    # A button inside a fragment normally reruns that fragment
                    # only.  The confirmation modal is rendered by main(), so
                    # switch to a full app rerun before opening it.
                    st.rerun(scope="app")

# =========================================================================== #
# Tab 3: Metrics
# =========================================================================== #
def page_metrics():
    try:
        data = api.get_metrics(ss.token)
    except APIError as e:
        st.error(str(e))
        return
    with st.container(key="pg_metrics"):
        st.markdown(ui.page_head("Giám sát vận hành RAG",
                                 "Dữ liệu thật của các truy vấn gần đây: ngữ cảnh truy xuất, độ trễ và lỗi xử lý."),
                    unsafe_allow_html=True)
        st.markdown(ui.kpis_html(data["kpis"]), unsafe_allow_html=True)
        st.markdown(ui.latency_html(data["latency"]), unsafe_allow_html=True)


@st.cache_data(ttl=2, show_spinner=False)
def cached_documents(token: str) -> list[dict]:
    """Poll documents only while the documents fragment is visible."""
    return api.list_documents(token)


@st.cache_data(ttl=90, show_spinner=False)
def cached_status(token: str) -> dict:
    return api.get_status(token)

# =========================================================================== #
# Tab 4: Cấu hình chỉ đọc
# =========================================================================== #
def page_config():
    # Read directly from the startup configuration.  Do not use session state:
    # users must not be able to change ingestion/index compatibility settings.
    cfg = api.DEFAULT_CONFIG
    with st.container(key="pg_cfg"):
        st.markdown(ui.readonly_config_html(cfg), unsafe_allow_html=True)

# =========================================================================== #
# Main
# =========================================================================== #
def main():
    authed = bool(ss.token)
    status = {"vector_db": "-", "total_vectors": 0, "embedding_model": "-",
              "collection": "-", "version": "v2.4"}
    if authed:
        try:
            status = cached_status(ss.token)
        except APIError as e:
            st.toast(str(e), icon="⚠️")

    up = [i for i, m in enumerate(ss.messages) if ss.fb.get(m.get("message_id")) == "up"]
    down = [i for i, m in enumerate(ss.messages) if ss.fb.get(m.get("message_id")) == "down"]
    st.markdown(ui.dynamic_css(authed, ss.sb_open, ss.tab, ss.cfg["search_mode"], ss.drawer_open,
                               ss.document_count,
                               up, down), unsafe_allow_html=True)

    if not authed:
        auth_page()
        return

    # tiêm JS vào trang cha 1 lần (tay nắm kéo độ rộng sidebar)
    run_js("<script>(function(){var p=window.parent;if(p.__sbInit)return;p.__sbInit=true;"
           "var s=p.document.createElement('script');s.textContent=" + json.dumps(SIDEBAR_JS) +
           ";p.document.head.appendChild(s);})();</script>")
    sidebar(status)
    if ss.get("show_logout_dialog"):
        logout_confirm_dialog()
    st.markdown(ui.header_html(TITLES[ss.tab], status["collection"], ss.cfg["search_mode"]),
                unsafe_allow_html=True)
    if ss.tab == "chat":
        page_chat()
    elif ss.tab == "documents":
        page_docs()
        if ss.get("delete_doc_to_confirm"):
            delete_document_confirm_dialog(ss.delete_doc_to_confirm)
    elif ss.tab == "config":
        page_config()
    else:
        page_metrics()


main()
