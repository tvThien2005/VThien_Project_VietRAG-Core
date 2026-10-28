"""
ui.py - CSS + HTML builders tái tạo giao diện mẫu (Tailwind 3 + Lucide).

Giá trị font/cỡ chữ/màu lấy đúng từ Tailwind:
  font-sans / font-mono mặc định của Tailwind; text-[10px] 10px, text-[11px] 11px, text-xs 12/16,
  text-sm 14/20, text-base 16/24, text-lg 18/28, text-2xl 24/32.
"""
import html
import re

import markdown

LUCIDE = "https://cdn.jsdelivr.net/npm/lucide-static@latest/icons/{}.svg"
SANS = ('ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", '
        'Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"')
MONO = ('ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace')

ICONS = ["cpu", "message-square", "database", "sliders-horizontal", "activity", "bell", "settings", "info", "bot",
         "circle-check", "circle-alert", "file-search", "paperclip", "send", "layers", "x", "file-text", "file-up",
         "upload", "link", "globe", "loader-circle", "scissors", "search", "thumbs-up", "thumbs-down", "plus",
         "trash-2", "log-out"]


def _u(n: str) -> str:
    return f'url("{LUCIDE.format(n)}")'


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def H(s: str) -> str:
    """Gộp HTML thành 1 dòng: tránh Markdown hiểu nhầm thụt lề/dòng trống thành code block."""
    return " ".join(l.strip() for l in s.strip().splitlines() if l.strip())


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def ic(name: str, size: int = 16, cls: str = "", color: str = "") -> str:
    style = f' style="color:{color}"' if color else ""
    return f'<span class="ic i-{name} s{size} {cls}"{style}></span>'


def fmt_size(n) -> str:
    if n is None:
        return "URL Scrape"
    return f"{n / 1048576:.1f} MB" if n >= 1048576 else f"{n / 1024:.0f} KB"


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
def _btn(sel: str, bg: str, color: str, border: str, hbg: str, hcolor: str | None = None) -> str:
    hc = hcolor or color
    return (f'{sel} button{{background:{bg}!important;color:{color}!important;border:{border}!important}}'
            f'{sel} button:hover,{sel} button:focus,{sel} button:active{{background:{hbg}!important;'
            f'color:{hc}!important;border:{border}!important;box-shadow:none!important;outline:none!important}}')


# selector -> icon (gắn bằng mask vào ::before của nút Streamlit)
ICON_BUTTONS = {
    ".st-key-nav_chat": "message-square", ".st-key-nav_documents": "database",
    ".st-key-nav_config": "sliders-horizontal", ".st-key-nav_metrics": "activity",
    ".st-key-ico_logout": "log-out", ".st-key-drawer_close": "x", ".st-key-drawer_reopen": "layers",
    '[class*="st-key-fbup_"]': "thumbs-up", '[class*="st-key-fbdn_"]': "thumbs-down",
    '[class*="st-key-ctx_"]': "file-search", ".st-key-hist_new": "plus", ".st-key-hist_del": "trash-2",
    ".st-key-pri_upload": "upload",
    ".st-key-ico_sb_close": "panel-left-close", ".st-key-ico_sb_open": "panel-left-open",
}
_ICON_SEL = ",".join(f"{s} button p::before" for s in ICON_BUTTONS)
_ICON_P_SEL = ",".join(f"{s} button p" for s in ICON_BUTTONS)

_STATIC = r"""
:root{--bg:#0b0f19;--card:#111827;--bd:#1f2937;--s100:#f1f5f9;--s200:#e2e8f0;--s300:#cbd5e1;--s400:#94a3b8;
--s500:#64748b;--s700:#334155;--s800:#1e293b;--s900:#0f172a;--s950:#020617;--i200:#c7d2fe;--i300:#a5b4fc;
--i400:#818cf8;--i500:#6366f1;--i600:#4f46e5;--i800:#3730a3;--i950:#1e1b4b;--em400:#34d399;--em800:#065f46;
--em950:#022c22;--am400:#fbbf24;--rd400:#f87171;--sbw:256px;color-scheme:dark;--sans:@SANS@;--mono:@MONO@}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--bg)!important;
color:var(--s200);-webkit-font-smoothing:antialiased}
*:not([data-testid="stIconMaterial"]){font-family:var(--sans)!important}
.stApp .mono,.stApp .mono *{font-family:var(--mono)!important}
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:#0b0f19}
::-webkit-scrollbar-thumb{background:#1f2937;border-radius:4px}::-webkit-scrollbar-thumb:hover{background:#374151}

/* ---- Streamlit chrome ---- */
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
#MainMenu,footer{display:none!important}
section[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="stExpandSidebarButton"],
[data-testid="stAppDeployButton"],[data-testid="stMainMenu"]{display:none!important}
[data-testid="stMainBlockContainer"],.block-container{padding:64px 0 0 var(--sbw)!important;max-width:100%!important}
.st-key-sb{position:fixed;top:0;left:0;bottom:0;width:var(--sbw);z-index:100;background:var(--card);
border-right:1px solid var(--bd);display:flex!important;flex-direction:column;justify-content:space-between;
overflow-x:hidden;overflow-y:auto;box-sizing:border-box}
#sb-handle{position:fixed;top:0;bottom:0;left:calc(var(--sbw) - 3px);width:6px;cursor:col-resize;z-index:1200;
transition:background .15s}
#sb-handle:hover,#sb-handle.on{background:rgba(99,102,241,.5)}
.st-key-ico_sb_close{position:absolute;top:16px;right:12px;z-index:5}
.st-key-ico_sb_open{position:fixed;top:16px;left:16px;z-index:120}
[data-testid="stVerticalBlock"]{gap:0!important}
[data-testid="stHorizontalBlock"]{gap:0!important;align-items:center}
[data-testid="stMarkdownContainer"] p{margin:0!important}
[data-testid="stWidgetLabel"] p{font-size:12px;line-height:16px;font-weight:500;color:var(--s300)}
[data-testid="InputInstructions"]{display:none}
/* Tắt hiệu ứng "mờ dần" (stale) của Streamlit khi script đang chạy lại: giao diện giữ nguyên độ sáng */
[data-stale="true"],[data-testid="stElementContainer"][data-stale="true"],.stale-element{opacity:1!important;
transition:none!important}
[data-testid="stSkeleton"]{display:none!important}

/* ---- icons ---- */
.ic{display:inline-block;flex-shrink:0;background-color:currentColor;-webkit-mask:var(--u) center/contain no-repeat;
mask:var(--u) center/contain no-repeat}
.s12{width:12px;height:12px}.s14{width:14px;height:14px}.s16{width:16px;height:16px}.s18{width:18px;height:18px}
.s20{width:20px;height:20px}.s24{width:24px;height:24px}
@keyframes pulse{50%{opacity:.5}}.pulse{animation:pulse 2s cubic-bezier(.4,0,.6,1) infinite}
@keyframes spin{to{transform:rotate(360deg)}}.spin{animation:spin 1s linear infinite}

/* ---- generic widgets ---- */
[data-baseweb="input"],[data-baseweb="textarea"]{background:var(--s900)!important;
border:1px solid var(--s800)!important;border-radius:8px!important}
[data-baseweb="base-input"]{background:transparent!important}
[data-baseweb="input"]:focus-within,[data-baseweb="textarea"]:focus-within{border-color:var(--i500)!important}
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea{background:transparent!important;
color:var(--s200)!important;font-size:12px;line-height:16px}
[data-baseweb="select"]>div{background:var(--s900)!important;border:1px solid var(--s800)!important;
border-radius:8px!important;min-height:0}
[data-baseweb="select"] *{font-size:12px}
[data-testid="stCheckbox"]{min-height:0}
[data-testid="stCheckbox"] label{gap:6px;align-items:center;min-height:0}
[data-testid="stCheckbox"] p{font-size:12px;line-height:16px;color:var(--s400)}
[data-baseweb="checkbox"]>span:first-of-type{width:16px;height:16px;border-radius:4px}
[data-baseweb="checkbox"]:has(input:checked)>span:first-of-type{background:var(--i600)!important;
border-color:var(--i600)!important}
[data-baseweb="checkbox"]:not(:has(input:checked))>span:first-of-type{background:var(--s800)!important;
border:1px solid var(--s700)!important}
[data-baseweb="select"] div{color:var(--s300)!important}
[data-testid="stSlider"] [role="slider"]{background:var(--i500)!important;border-color:var(--i500)!important}
[data-testid="stSliderThumbValue"],[data-testid="stTickBar"]{display:none}
[data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--s800)}
button[role="tab"] p{font-size:14px;line-height:20px;font-weight:500;color:var(--s400)}
button[role="tab"][aria-selected="true"] p{color:var(--i400)}
[data-baseweb="tab-highlight"]{background:var(--i500)!important}
[data-baseweb="tab-border"]{display:none}
[data-testid="stButton"] button,[data-testid="stFormSubmitButton"] button{background:var(--s800);color:var(--s300);
border:1px solid var(--s700);border-radius:8px;font-size:12px;line-height:16px;font-weight:500;min-height:0;
box-shadow:none;padding:6px 12px}
[data-testid="stButton"] button p,[data-testid="stFormSubmitButton"] button p{margin:0;font-size:inherit;
line-height:inherit}
[data-testid="stElementContainer"]:has(>[data-testid="stButton"]){width:auto}

/* ---- sidebar ---- */
.logo{height:64px;display:flex;align-items:center;gap:12px;padding:0 24px;border-bottom:1px solid var(--bd)}
.st-key-sb .logo{padding:0 56px 0 24px}.logo>div:last-child{min-width:0}.logo .n,.logo .s{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.logo .box{padding:8px;background:rgba(79,70,229,.2);color:var(--i400);border-radius:8px;display:flex}
.logo .n{font-size:14px;line-height:20px;font-weight:700;letter-spacing:.025em;color:var(--s100)}
.logo .s{font-size:12px;line-height:16px;color:var(--s400)}
.st-key-nav{padding:16px}
.st-key-nav [data-testid="stElementContainer"]+[data-testid="stElementContainer"]{margin-top:4px}
[class*="st-key-nav_"] [data-testid="stButton"],[class*="st-key-nav_"] button{width:100%}
[class*="st-key-nav_"] button{justify-content:flex-start;padding:12px 16px;border-radius:8px;font-size:14px;
line-height:20px;font-weight:500;transition:all .15s}
[class*="st-key-nav_"] button>div,[class*="st-key-nav_"] button p{width:100%}
[class*="st-key-nav_"] button p{display:flex;align-items:center;gap:12px;font-size:14px;line-height:20px;
font-weight:500;text-align:left;white-space:nowrap}
[class*="st-key-nav_"] button p::before{width:18px;height:18px}
.st-key-nav_documents button p::after{margin-left:auto;background:var(--i950);color:var(--i300);font-size:12px;
line-height:16px;font-weight:400;padding:2px 8px;border-radius:9999px;border:1px solid var(--i800)}
.st-key-side_bot{padding:16px;border-top:1px solid var(--bd)}
.stat{background:rgba(15,23,42,.8);border:1px solid var(--s800);border-radius:8px;padding:12px;font-size:12px;
line-height:16px;margin-bottom:12px}
.stat .r{display:flex;align-items:center;justify-content:space-between}.stat .r+.r{margin-top:8px}
.stat .k{color:var(--s400)}.stat .v{color:var(--s200)}
.dot{width:6px;height:6px;border-radius:9999px;background:var(--em400);display:inline-block}
.me{display:flex;align-items:center;gap:12px;overflow:hidden}.st-key-side_user{padding-top:8px}
.me .av2{width:32px;height:32px;border-radius:9999px;background:rgba(79,70,229,.3);
border:1px solid rgba(99,102,241,.5);display:flex;align-items:center;justify-content:center;font-weight:700;
font-size:12px;line-height:16px;color:var(--i300);flex-shrink:0}
.me .nm{font-size:12px;line-height:16px;font-weight:500;color:var(--s200);white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.me .em{font-size:11px;line-height:16px;color:var(--s400);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.me>div:last-child{min-width:0}
[class*="st-key-ico_"] button,[class*="st-key-drawer_close"] button,[class*="st-key-hist_del"] button{width:32px;
height:32px;padding:0;justify-content:center;border-radius:8px}
[class*="st-key-ico_"] button p,.st-key-drawer_close button p,.st-key-hist_del button p{font-size:0}

/* ---- header ---- */
.hdr{position:fixed;top:0;left:var(--sbw);right:0;height:64px;z-index:90;display:flex;align-items:center;
justify-content:space-between;padding:0 24px;border-bottom:1px solid var(--bd);background:#0e1420}
.hdr .l{display:flex;align-items:center;gap:16px}.hdr .r{display:flex;align-items:center;gap:12px}
.hdr .t{font-size:16px;line-height:24px;font-weight:600;color:var(--s100)}
.chip{background:var(--s800);color:var(--s300);font-size:12px;line-height:16px;padding:4px 10px;border-radius:6px;
border:1px solid var(--s700)}.chip strong{color:var(--i400);font-weight:700}
.mode{display:flex;align-items:center;background:var(--s900);border:1px solid var(--s800);border-radius:8px;
padding:4px;font-size:12px;line-height:16px}
.mode .m1{padding:4px 10px;color:var(--s400)}
.mode .m2{padding:4px 10px;background:rgba(79,70,229,.3);color:var(--i300);border-radius:4px;font-weight:500;
border:1px solid rgba(99,102,241,.3);text-transform:uppercase;letter-spacing:.05em;font-size:10px}
.ibtn{padding:8px;border-radius:8px;background:var(--s800);color:var(--s300);display:flex}

/* ---- chat ---- */
.st-key-chatpane{padding:24px 24px 170px}
.st-key-histbar{max-width:768px;margin-bottom:24px}
.st-key-histbar [data-testid="stHorizontalBlock"]{gap:8px}
.banner{display:flex;align-items:flex-start;gap:12px;background:rgba(30,27,75,.4);
border:1px solid rgba(55,48,163,.5);border-radius:12px;padding:16px;margin-bottom:24px}
.banner .ic{color:var(--i400);margin-top:2px}.banner .tx{font-size:12px;line-height:16px}
.banner .tx div+div{margin-top:4px}.banner .a{color:var(--i200);font-weight:500}
.banner .b{color:rgba(165,180,252,.7)}
.ub-wrap{display:flex;justify-content:flex-end;margin-bottom:24px}
.ub{background:var(--i600);color:#fff;border-radius:16px 0 16px 16px;padding:14px 20px;max-width:672px;
font-size:14px;line-height:20px;box-shadow:0 10px 15px -3px rgba(79,70,229,.1),0 4px 6px -4px rgba(79,70,229,.1)}
.ub .t{display:block;text-align:right;font-size:10px;line-height:15px;color:var(--i200);margin-top:4px}
.ai{display:flex;gap:16px;max-width:768px}
.av{width:32px;height:32px;border-radius:12px;background:linear-gradient(to bottom right,#6366f1,#9333ea);
display:flex;align-items:center;justify-content:center;color:#fff;flex-shrink:0;
box-shadow:0 4px 6px -1px rgba(0,0,0,.1),0 2px 4px -2px rgba(0,0,0,.1)}
.ai-b{min-width:0;flex:1}
.ab{background:var(--s900);border:1px solid var(--s800);border-radius:0 16px 16px 16px;padding:20px;
font-size:14px;line-height:20px;color:var(--s200);box-shadow:0 1px 2px 0 rgba(0,0,0,.05)}
.ab>*+*{margin-top:12px}
.ans>*+*{margin-top:12px}.ans p{margin:0}
.ans ul{margin:0;padding-left:20px;list-style:disc;color:var(--s300)}.ans li+li{margin-top:8px}
.ans strong{font-weight:700}
.note{font-size:12px;line-height:16px;color:var(--s400);padding-top:8px;border-top:1px solid var(--s800)}
.meta{display:flex;align-items:center;gap:16px;font-size:11px;line-height:16px;color:var(--s400);padding:0 4px;
margin-top:12px}
.meta .g{display:flex;align-items:center;gap:4px;color:var(--em400)}.meta .g.bad{color:var(--am400)}
.meta strong{font-weight:700}
.cite{display:inline-flex;align-items:center;gap:4px;font-size:11px;line-height:16px;background:var(--i950);
color:var(--i400);border:1px solid rgba(55,48,163,.8);border-radius:4px;padding:2px 6px;margin-left:4px}
[class*="st-key-rowr_"],[class*="st-key-rowf_"]{display:flex!important;flex-direction:row!important;
justify-content:flex-end;align-items:center;gap:12px}
[class*="st-key-rowr_"] [data-testid="stElementContainer"],
[class*="st-key-rowf_"] [data-testid="stElementContainer"]{width:auto!important}
[class*="st-key-rowr_"]{max-width:768px;padding:0 4px 0 48px;box-sizing:border-box;margin:-17px 0 24px;height:17px}
[class*="st-key-rowf_"]{gap:8px}
[class*="st-key-fbup_"] button,[class*="st-key-fbdn_"] button{width:20px;height:17px;padding:0;justify-content:center;
border-radius:4px}
[class*="st-key-fbup_"] button p,[class*="st-key-fbdn_"] button p{font-size:0}
[class*="st-key-fbup_"] button p::before,[class*="st-key-fbdn_"] button p::before,
[class*="st-key-ctx_"] button p::before{width:14px;height:14px}
[class*="st-key-ctx_"] button{padding:0}
[class*="st-key-ctx_"] button p{display:flex;align-items:center;gap:4px;font-size:11px;line-height:16px;
font-weight:400}
[class*="st-key-ctx_"] button:hover p{text-decoration:underline}
.st-key-dock{position:fixed;bottom:0;left:var(--sbw);right:0;width:auto!important;box-sizing:border-box;z-index:80;background:#0d121d;
border-top:1px solid var(--bd);padding:16px}
.st-key-dock_in{width:100%;max-width:896px;margin:0 auto;box-sizing:border-box}
.st-key-dockrow{padding:0 8px;margin-bottom:8px}
.topk{text-align:right;font-size:12px;line-height:16px;color:var(--s400)}
.topk strong{color:var(--i400);font-weight:700}
.st-key-formbox{position:relative;--pc:@PAPERCLIP@;--send:@SEND@}
.st-key-formbox [data-testid="stForm"]{border:0;padding:0;background:transparent}
.st-key-formbox [data-baseweb="textarea"]{border-radius:12px!important}
.st-key-formbox textarea{padding:12px 112px 12px 16px!important;font-size:14px!important;line-height:20px!important;
resize:none;box-shadow:inset 0 2px 4px 0 rgba(0,0,0,.05)}
.st-key-formbox::before{content:"";position:absolute;right:64px;top:26px;width:16px;height:16px;z-index:3;
pointer-events:none;background:var(--s400);-webkit-mask:var(--pc) center/contain no-repeat;
mask:var(--pc) center/contain no-repeat}
.st-key-chat_file_upload{position:absolute;right:54px;top:16px;z-index:4;width:36px;height:36px}
.st-key-chat_file_upload [data-testid="stFileUploader"]{position:absolute;inset:0;width:36px!important;height:36px!important;
opacity:0;cursor:pointer}
.st-key-chat_file_upload [data-testid="stFileUploader"] section,
.st-key-chat_file_upload [data-testid="stFileUploaderDropzone"]{min-height:36px!important;height:36px!important;
padding:0!important;cursor:pointer}
.st-key-send_box{position:absolute;right:12px;top:16px;z-index:5;width:36px;height:36px}
.st-key-send_box button{width:36px;height:36px;padding:0;border-radius:8px;justify-content:center}
.st-key-send_box button p{font-size:0}
.st-key-send_box button p::before{content:"";display:block;width:16px;height:16px;
background:currentColor;-webkit-mask:var(--send) center/contain no-repeat;mask:var(--send) center/contain no-repeat}
.drawer{position:fixed;top:64px;right:0;bottom:0;width:384px;background:var(--card);border-left:1px solid var(--bd);
display:flex;flex-direction:column;z-index:70}
.drawer .dh{padding:16px;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:8px}
.drawer .dh .ic{color:var(--i400)}
.drawer .dh .t{font-size:14px;line-height:20px;font-weight:600;color:var(--s200)}
.drawer .db{flex:1;overflow-y:auto;padding:16px}.drawer .db>*+*{margin-top:16px}
.drawer .empty{font-size:12px;line-height:16px;color:var(--s500)}
.chunk{background:rgba(15,23,42,.5);border:1px solid var(--s800);border-radius:12px;padding:14px}
.chunk>*+*{margin-top:8px}
.chunk.hl{background:rgba(15,23,42,.9);border-color:rgba(99,102,241,.4);
box-shadow:0 4px 6px -1px rgba(0,0,0,.1),0 2px 4px -2px rgba(0,0,0,.1)}
.chunk.dim{opacity:.6}
.chunk .r1{display:flex;justify-content:space-between;align-items:center;font-size:12px;line-height:16px}
.chunk .id{color:var(--s400);background:var(--s800);padding:2px 8px;border-radius:4px}
.chunk.hl .id{color:var(--i400);background:var(--i950);border:1px solid var(--i800)}
.chunk .sc{font-weight:500;color:var(--em400)}.chunk .sc.low{color:var(--am400)}
.chunk .tx{font-size:12px;line-height:1.625;color:var(--s300);background:rgba(2,6,23,.5);padding:10px;
border-radius:4px;border:1px solid rgba(30,41,59,.8);max-height:240px;overflow-y:auto;overflow-wrap:anywhere;
white-space:pre-wrap;scrollbar-gutter:stable}
.chunk.dim .tx{color:var(--s400)}
.chunk .hist-src,.chunk .sec{font-size:11px;line-height:16px;color:var(--s500)}
.chunk .sec{color:var(--i300);overflow-wrap:anywhere}
.chunk .r3{display:flex;justify-content:space-between;align-items:center;font-size:11px;line-height:16px;
color:var(--s400);padding-top:4px}
.chunk .fn{display:flex;align-items:center;gap:4px;max-width:180px;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap}.chunk .fn .ic{color:var(--s500)}
.drawer .df{padding:16px;border-top:1px solid var(--bd);background:var(--s950);font-size:12px;line-height:16px}
.drawer .df .r{display:flex;justify-content:space-between;font-weight:600;color:var(--s300)}
.drawer .df .r span:last-child{color:var(--i400);font-size:10px;font-weight:400}
.drawer .df .q{margin-top:8px;color:var(--s400);font-size:11px;font-style:italic;background:var(--s900);
padding:8px;border-radius:4px;border:1px solid var(--s800)}
.st-key-drawer_close{position:fixed;top:82px;right:16px;z-index:1100}
.st-key-drawer_close button{width:16px;height:16px;padding:0}
.st-key-drawer_reopen{position:fixed;top:82px;right:16px;z-index:1100}
.st-key-drawer_reopen button{padding:8px 12px;border-radius:8px;font-size:12px;line-height:16px}

/* ---- pages ---- */
.st-key-pg_docs,.st-key-pg_metrics{padding:24px}
.st-key-pg_cfg{padding:24px;max-width:944px}
.cfg-ro{max-width:896px}.cfg-ro .ph{margin-bottom:16px}.cfg-ro-note{display:flex;gap:10px;align-items:flex-start;
background:rgba(30,58,95,.62);border:1px solid rgba(59,130,246,.24);border-radius:10px;padding:12px 14px;
margin-bottom:20px;color:#93c5fd;font-size:12px;line-height:18px}.cfg-ro-note .ico{font-size:15px;line-height:18px}
.cfg-ro-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.cfg-ro-panel{background:var(--card);
border:1px solid var(--bd);border-radius:12px;padding:18px;margin-top:16px}.cfg-ro-panel:first-of-type{margin-top:0}
.cfg-ro-panel-title{display:flex;align-items:center;gap:8px;padding-bottom:12px;margin-bottom:14px;border-bottom:1px solid var(--s800);
font-size:14px;line-height:20px;font-weight:600;color:var(--s200)}.cfg-ro-panel-title .ic{color:var(--i400)}
.cfg-ro-item{min-width:0;background:var(--s900);border:1px solid var(--s800);border-radius:8px;padding:12px}
.cfg-ro-item.wide{grid-column:1/-1}.cfg-ro-k{font-size:11px;line-height:16px;color:var(--s400);margin-bottom:5px}
.cfg-ro-v{font-size:13px;line-height:20px;font-weight:500;color:var(--s200);overflow-wrap:anywhere}
.cfg-ro-v.num{font-family:var(--mono)!important;font-size:22px;line-height:28px;color:var(--i300)}
.cfg-ro-v.status{display:inline-flex;align-items:center;gap:5px;color:var(--em400)}.cfg-ro-v.status.off{color:var(--s400)}
.cfg-ro-foot{font-size:11px;line-height:16px;color:var(--s500);margin:14px 2px 0}
@media(max-width:760px){.cfg-ro-grid{grid-template-columns:1fr}.cfg-ro-item.wide{grid-column:auto}}
.ph .t{font-size:18px;line-height:28px;font-weight:700;color:var(--s100)}
.ph .d{font-size:12px;line-height:16px;color:var(--s400)}
[class*="sec_"] button,[class*="pri_"] button{padding:8px 16px;border-radius:8px;font-size:14px;line-height:20px;
font-weight:500;gap:8px}
[class*="sec_"] button p,[class*="pri_"] button p{display:flex;align-items:center;gap:8px;font-size:14px;
line-height:20px}
.st-key-pri_upload button p::before{width:16px;height:16px}
.st-key-pri_upload button,.st-key-pri_save button{box-shadow:0 10px 15px -3px rgba(79,70,229,.2),
0 4px 6px -4px rgba(79,70,229,.2)!important}
.st-key-sec_reset button,.st-key-pri_save button{font-size:12px;line-height:16px;padding:10px 20px;border:0}
.st-key-pri_save button{padding:10px 24px}
.st-key-sec_reset button p,.st-key-pri_save button p{font-size:12px;line-height:16px}
.st-key-dz{position:relative;margin-top:24px}
.dz{border:2px dashed var(--s800);border-radius:16px;padding:32px;text-align:center;background:rgba(17,24,39,.4);
transition:border-color .15s}
.st-key-dz:hover .dz{border-color:rgba(99,102,241,.5)}
.dz .ib{width:48px;height:48px;background:rgba(79,70,229,.1);color:var(--i400);border-radius:12px;display:flex;
align-items:center;justify-content:center;margin:0 auto 12px;transition:transform .15s}
.st-key-dz:hover .dz .ib{transform:scale(1.1)}
.dz .a{font-size:14px;line-height:20px;font-weight:500;color:var(--s200)}
.dz .b{font-size:12px;line-height:16px;color:var(--s500);margin-top:4px}
.st-key-dz [data-testid="stFileUploader"]{position:absolute;top:0;left:0;width:100%;height:100%;opacity:0}
.st-key-dz [data-testid="stFileUploader"] section{height:100%;cursor:pointer}
.st-key-dz [data-testid="stFileUploaderDropzone"]{height:100%;cursor:pointer}
.st-key-tbl{background:var(--card);border:1px solid var(--bd);border-radius:12px;overflow:hidden;margin-top:24px}
.st-key-tbl_bar{padding:16px;border-bottom:1px solid var(--bd)}
.st-key-tbl_q{width:256px}
.st-key-tbl_q input{padding:6px 12px!important}
.cnt{text-align:right;font-size:12px;line-height:16px;color:var(--s400)}
.st-key-tbl_head{background:rgba(15,23,42,.8);border-bottom:1px solid var(--s800)}
.th{padding:16px;font-size:10px;line-height:15px;text-transform:uppercase;letter-spacing:.05em;color:var(--s400)}
[class*="st-key-trow_"]{border-bottom:1px solid rgba(30,41,59,.6)}
[class*="st-key-trow_"]:hover{background:rgba(30,41,59,.3)}
.td{padding:16px;font-size:12px;line-height:16px;color:var(--s300)}
.td.nm2{display:flex;align-items:center;gap:10px;font-weight:500;color:var(--s200);word-break:break-all}
.pill{display:flex;align-items:center;gap:4px;width:max-content;padding:2px 8px;border-radius:9999px;
font-size:10px;line-height:15px;font-weight:500;border:1px solid}
.pill.ok{background:var(--em950);color:var(--em400);border-color:var(--em800)}
.pill.run{background:var(--i950);color:var(--i400);border-color:var(--i800)}
.pill.err{background:#450a0a;color:var(--rd400);border-color:#991b1b}
[class*="st-key-lnk_"] button,[class*="st-key-dng_"] button{padding:0;font-size:12px;line-height:16px;
font-weight:400;border-radius:0}
[class*="st-key-lnk_"] button:disabled{color:var(--s500)!important;cursor:not-allowed}
[class*="st-key-rowf_a_"]{padding-right:16px}
[class*="st-key-card_"]{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:20px;
margin-top:24px}
[class*="st-key-card_"] [data-testid="stHorizontalBlock"]{gap:24px;align-items:flex-start}
.st-key-cfg_row2{margin-top:24px}
.ch{display:flex;align-items:center;gap:8px;font-size:14px;line-height:20px;font-weight:600;color:var(--s200);
border-bottom:1px solid var(--s800);padding-bottom:8px;margin-bottom:16px}.ch .ic{color:var(--i400)}
.lb{font-size:12px;line-height:16px;font-weight:500;color:var(--s300);margin-bottom:6px}
.lb .v{color:var(--i400);font-weight:400}
.ticks{display:flex;justify-content:space-between;font-size:10px;line-height:15px;color:var(--s500)}
.st-key-mode_row [data-testid="stHorizontalBlock"]{gap:8px}
.st-key-rr_row{background:rgba(15,23,42,.6);border:1px solid var(--s800);border-radius:8px;padding:12px;
margin-top:20px}
.rrt .a{font-size:12px;line-height:16px;font-weight:500;color:var(--s200)}
.rrt .b{font-size:11px;line-height:16px;color:var(--s400)}
.st-key-cfg_foot{margin-top:24px}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-top:24px}
.kpi{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px}
.kpi>*+*{margin-top:4px}
.kpi .l{font-size:12px;line-height:16px;color:var(--s400)}
.kpi .vr{display:flex;align-items:baseline;gap:8px}
.kpi .v{font-size:24px;line-height:32px;font-weight:700}.kpi .dl{font-size:10px;line-height:15px}
.kpi .d{font-size:10px;line-height:15px;color:var(--s500)}
.lat{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:20px;margin-top:24px}
.lat .h{font-size:14px;line-height:20px;font-weight:600;color:var(--s200)}
.lat .its{margin-top:16px;font-size:12px;line-height:16px}.lat .it+.it{margin-top:12px}
.lat .it .t{display:flex;justify-content:space-between;color:var(--s400);margin-bottom:4px}
.bar{width:100%;background:var(--s900);border-radius:9999px;height:8px;overflow:hidden}
.bar i{display:block;height:8px;border-radius:9999px}
.st-key-auth{max-width:400px;margin:96px auto 0;background:var(--card);border:1px solid var(--bd);
border-radius:16px;padding:32px}
.st-key-auth .logo{height:auto;padding:0 0 24px;border:0}
.st-key-auth [data-testid="stFormSubmitButton"] button{width:100%;background:var(--i600)!important;color:#fff!important;
border:0!important;padding:10px 16px;font-size:14px;line-height:20px;margin-top:8px}
.st-key-auth [data-testid="stForm"]{border:0;padding:16px 0 0}
"""

_NAV_ICON_CSS = (f"{_ICON_SEL}{{content:\"\";display:block;flex-shrink:0;background:currentColor;"
                 f"-webkit-mask:var(--ico) center/contain no-repeat;mask:var(--ico) center/contain no-repeat;"
                 f"width:16px;height:16px}}"
                 f"{_ICON_P_SEL}{{display:flex;align-items:center;gap:8px}}"
                 + "".join(f"{s}{{--ico:{_u(n)}}}" for s, n in ICON_BUTTONS.items()))

_BUTTON_COLORS = (
    _btn('[class*="st-key-nav_"]', "transparent", "var(--s400)", "0", "rgba(30,41,59,.5)", "var(--s200)")
    + _btn('[class*="sec_"]', "var(--s800)", "var(--s200)", "1px solid var(--s700)", "var(--s700)")
    + _btn(".st-key-sec_reset", "var(--s800)", "var(--s300)", "0", "var(--s700)")
    + _btn('[class*="pri_"]', "var(--i600)", "#fff", "0", "var(--i500)")
    + _btn('[class*="st-key-lnk_"]', "transparent", "var(--i400)", "0", "transparent", "var(--i300)")
    + _btn('[class*="st-key-dng_"]', "transparent", "var(--rd400)", "0", "transparent", "#fca5a5")
    + _btn('[class*="st-key-ico_"]', "transparent", "var(--s400)", "0", "var(--s800)", "var(--s200)")
    + _btn(".st-key-drawer_close", "transparent", "var(--s400)", "0", "transparent", "var(--s200)")
    + _btn(".st-key-drawer_reopen", "var(--i600)", "#fff", "0", "var(--i500)")
    + _btn('[class*="st-key-fbup_"]', "transparent", "var(--s500)", "0", "transparent", "var(--em400)")
    + _btn('[class*="st-key-fbdn_"]', "transparent", "var(--s500)", "0", "transparent", "var(--am400)")
    + _btn('[class*="st-key-ctx_"]', "transparent", "var(--i400)", "0", "transparent", "var(--i400)")
    + _btn(".st-key-hist_new", "var(--s800)", "var(--s300)", "1px solid var(--s700)", "var(--s700)")
    + _btn(".st-key-hist_del", "var(--s800)", "var(--s300)", "1px solid var(--s700)", "var(--s700)")
    + _btn('[class*="st-key-mode_"]', "var(--s900)", "var(--s400)", "1px solid var(--s800)", "var(--s800)")
    + _btn(".st-key-send_box", "var(--i600)", "#fff", "0", "var(--i500)")
    + _btn(".st-key-ico_sb_open", "var(--s800)", "var(--s300)", "0", "var(--s700)", "var(--s200)")
)
_MODE_BTN = ('[class*="st-key-mode_"] button{width:100%;padding:8px;border-radius:8px;font-size:12px;'
             'line-height:16px;font-weight:500;justify-content:center}')

BASE_CSS = re.sub(
    r"\s*\n\s*", "",
    _STATIC.replace("@SANS@", SANS).replace("@MONO@", MONO)
    .replace("@PAPERCLIP@", _u("paperclip")).replace("@SEND@", _u("send"))
) + "".join(f".i-{n}{{--u:{_u(n)}}}" for n in ICONS) + _NAV_ICON_CSS + _BUTTON_COLORS + _MODE_BTN


def dynamic_css(authed: bool, sb_open: bool, tab: str, mode: str, drawer_open: bool, doc_count: int,
                fb_up: list[int], fb_down: list[int]) -> str:
    c = []
    if not authed:
        c.append(':root{--sbw:0px!important}.st-key-sb,#sb-handle,.st-key-dock{display:none!important}'
                 '[data-testid="stMainBlockContainer"],.block-container{padding-top:0!important}')
    else:
        if not sb_open:
            c.append(':root{--sbw:0px!important}.st-key-sb,#sb-handle{display:none!important}'
                     '.hdr{padding-left:64px!important}')
        a = f".st-key-nav_{tab} button"
        c.append(f'{a},{a}:hover,{a}:focus{{background:rgba(79,70,229,.1)!important;color:var(--i400)!important;'
                 f'border-right:2px solid var(--i500)!important}}')
        c.append(f'.st-key-nav_documents button p::after{{content:"{doc_count}"}}')
        m = f".st-key-mode_{mode} button"
        c.append(f'{m},{m}:hover,{m}:focus{{background:var(--i600)!important;border:1px solid var(--i500)!important;'
                 f'color:#fff!important}}')
        if tab == "chat" and drawer_open:
            c.append('.st-key-dock{right:384px!important}.st-key-chatpane{padding-right:408px!important}')
        for i in fb_up:
            c.append(f'.st-key-fbup_{i} button{{color:var(--em400)!important}}')
        for i in fb_down:
            c.append(f'.st-key-fbdn_{i} button{{color:var(--am400)!important}}')
    return "<style>" + BASE_CSS + "".join(c) + "</style>"


# --------------------------------------------------------------------------- #
# HTML builders
# --------------------------------------------------------------------------- #
def logo_html(version: str = "v2.4") -> str:
    return H(f'<div class="logo"><div class="box">{ic("cpu", 24)}</div><div><div class="n">NexusRAG AI</div>'
             f'<div class="s">Enterprise Engine {esc(version)}</div></div></div>')


def status_html(s: dict) -> str:
    return H(f'''<div class="stat">
    <div class="r"><span class="k">Vector DB:</span><span class="v mono" style="color:var(--em400);display:flex;align-items:center;gap:4px"><span class="dot pulse"></span> {esc(s["vector_db"])}</span></div>
    <div class="r"><span class="k">Total Vectors:</span><span class="v mono">{int(s["total_vectors"]):,}</span></div>
    <div class="r"><span class="k">Embedding Model:</span><span class="v mono" style="font-size:10px">{esc(s["embedding_model"])}</span></div>
    </div>''')


def user_html(u: dict) -> str:
    name = u.get("full_name") or u.get("email", "?")
    ini = "".join(w[0] for w in name.split()[:2]).upper() or "U"
    return H(f'<div class="me"><div class="av2">{esc(ini)}</div><div><div class="nm">{esc(name)}</div>'
             f'<div class="em">{esc(u.get("email", ""))}</div></div></div>')


def header_html(title: str, collection: str, mode: str) -> str:
    return H(f'<div class="hdr"><div class="l"><div class="t">{esc(title)}</div><span class="chip">Collection: '
             f'<strong>{esc(collection)}</strong></span></div><div class="r"><div class="mode"><span class="m1">'
             f'Search Mode:</span><span class="m2">{esc(mode)}</span></div><span class="ibtn">{ic("bell")}</span>'
             f'<span class="ibtn">{ic("settings")}</span></div></div>')


def page_head(title: str, desc: str) -> str:
    return H(f'<div class="ph"><div class="t">{esc(title)}</div><div class="d">{esc(desc)}</div></div>')


def readonly_config_html(cfg: dict) -> str:
    """A self-contained read-only configuration view, insulated from widget CSS."""
    rerank_enabled = bool(cfg.get("rerank"))
    rerank_text = "Bật" if rerank_enabled else "Tắt"
    rerank_class = "status" if rerank_enabled else "status off"
    return H(f'''<section class="cfg-ro">
        {page_head("Cấu hình RAG & Retrieval Engine", "Chế độ chỉ đọc · Cấu hình hệ thống được quản trị trong configs/config.yaml.")}
        <div class="cfg-ro-note"><span class="ico">🔒</span><span>Các thay đổi về chunking, embedding hoặc retrieval có thể yêu cầu ingest/index lại tài liệu. Liên hệ quản trị viên để thay đổi cấu hình hệ thống.</span></div>
        <div class="cfg-ro-panel"><div class="cfg-ro-panel-title">{ic("scissors")} Chunking Strategy (Phân đoạn văn bản)</div>
            <div class="cfg-ro-grid">
                <div class="cfg-ro-item"><div class="cfg-ro-k">Phương pháp chia đoạn</div><div class="cfg-ro-v">{esc(cfg.get("splitter", "-"))}</div></div>
                <div class="cfg-ro-item"><div class="cfg-ro-k">Chunk size</div><div class="cfg-ro-v num">{esc(cfg.get("chunk_size", "-"))} tokens</div></div>
                <div class="cfg-ro-item"><div class="cfg-ro-k">Chunk overlap</div><div class="cfg-ro-v num">{esc(cfg.get("chunk_overlap", "-"))} tokens</div></div>
                <div class="cfg-ro-item wide"><div class="cfg-ro-k">Embedding model</div><div class="cfg-ro-v mono">{esc(cfg.get("embedding", "-"))}</div></div>
            </div>
        </div>
        <div class="cfg-ro-panel"><div class="cfg-ro-panel-title">{ic("search")} Retrieval & Search Strategy</div>
            <div class="cfg-ro-grid">
                <div class="cfg-ro-item"><div class="cfg-ro-k">Chế độ truy vấn</div><div class="cfg-ro-v mono">{esc(cfg.get("search_mode", "-"))}</div></div>
                <div class="cfg-ro-item"><div class="cfg-ro-k">Top-K contexts</div><div class="cfg-ro-v num">{esc(cfg.get("top_k", "-"))}</div></div>
                <div class="cfg-ro-item"><div class="cfg-ro-k">Re-ranking</div><div class="cfg-ro-v {rerank_class}">{rerank_text}</div></div>
            </div>
        </div>
        <div class="cfg-ro-foot">Giá trị được nạp khi Streamlit khởi động. Sau khi quản trị viên đổi config.yaml, hãy restart API và Streamlit.</div>
    </section>''')


def banner_html() -> str:
    return H(f'''<div class="banner">{ic("info", 20)}<div class="tx"><div class="a">Chế độ RAG Grounding đang hoạt động</div>
    <div class="b">Mọi câu trả lời được kiểm tra đối soát trực tiếp từ tài liệu của bạn. Các đoạn văn bản trích dẫn sẽ hiển thị chi tiết ở khung bên phải.</div></div></div>''')


def user_bubble(m: dict) -> str:
    text = esc(m["content"]).replace("\n", "<br>")
    return H(f'<div class="ub-wrap"><div class="ub"><div>{text}</div><span class="t">{esc(m.get("ts", ""))}</span></div></div>')


def answer_html(text: str, sources: list[dict]) -> str:
    body = markdown.markdown(html.escape(text or "", quote=False), extensions=["sane_lists"])

    def pill(mt):
        n = int(mt.group(1))
        if 1 <= n <= len(sources):
            return f'<span class="cite mono">[Nguồn {n}: Score {float(sources[n - 1]["score"]):.2f}]</span>'
        return mt.group(0)

    return re.sub(r"\[(\d+)\]", pill, body)


def meta_html(mt: dict) -> str:
    if mt.get("grounded", True):
        g = f'<span class="g">{ic("circle-check", 14)} Grounded</span>'
    else:
        g = f'<span class="g bad">{ic("circle-alert", 14)} Not grounded</span>'
    return (f'<div class="meta mono">{g}<span>• Total Latency: <strong>{mt.get("total_ms", "-")}ms</strong></span>'
            f'<span>• Retrieval: <strong>{mt.get("retrieval_ms", "-")}ms</strong></span>'
            f'<span>• Contexts used: <strong>{mt.get("contexts_used", "-")}/{mt.get("contexts_total", "-")} '
            f'chunks</strong></span></div>')


def ai_message(m: dict) -> str:
    src = m.get("sources") or []
    note = f'<div class="note"><em>Lưu ý: {esc(m["note"])}</em></div>' if m.get("note") else ""
    meta = meta_html(m["metrics"]) if m.get("metrics") else ""
    return H(f'<div class="ai"><div class="av">{ic("bot", 20)}</div><div class="ai-b"><div class="ab">'
             f'<div class="ans">{answer_html(m["content"], src)}</div>{note}</div>{meta}</div></div>')


def thinking_html() -> str:
    return H(f'<div class="ai"><div class="av">{ic("bot", 20)}</div><div class="ai-b"><div class="ab" '
             f'style="display:flex;align-items:center;gap:8px;color:var(--s400)">{ic("loader-circle", 16, "spin")} '
             f'Đang truy xuất và sinh câu trả lời...</div></div></div>')


def chunk_card(i: int, s: dict, hl: bool = False) -> str:
    used = s.get("used", True)
    cls = "chunk" + (" hl" if hl else "") + ("" if used else " dim")
    try:
        score = float(s.get("score", 0) or 0)
    except (TypeError, ValueError):
        score = 0.0
    low = " low" if score < 0.7 else ""
    icon = ic("file-text", 14) if used else ""
    history_notice = "" if s.get("text") else "<div class=\"hist-src\">Nội dung chunk không được lưu trong lịch sử.</div>"
    section = (f'<div class="sec">{esc(s.get("section_path"))}</div>' if s.get("section_path") else "")
    return (f'<div class="{cls}"><div class="r1"><span class="id mono">Chunk #{i}</span>'
            f'<span class="sc mono{low}">Similarity: {score:.3f}</span></div>'
            f'<div class="tx" tabindex="0" title="Cuộn để xem toàn bộ nội dung chunk">"{esc(s.get("text", ""))}"</div>{history_notice}{section}'
            f'<div class="r3"><span class="fn" title="{esc(s["doc_name"])}">'
            f'{icon}{esc(s.get("doc_name", "Không rõ tài liệu"))}</span><span>Trang {esc(s.get("page", "?"))}</span></div></div>')


def drawer_html(m: dict | None, use_hybrid: bool = False) -> str:
    src = (m or {}).get("sources") or []
    cards = "".join(chunk_card(i, s, hl=(i == 1)) for i, s in enumerate(src, 1)) or \
        '<div class="empty">Chưa có context nào. Hãy đặt câu hỏi để xem các đoạn được truy xuất.</div>'
    rq = (m or {}).get("rewritten_query") or "—"
    mode = "Hybrid Search" if use_hybrid else "Dense Vector"
    return H(f'<div class="drawer"><div class="dh">{ic("layers")}<div class="t">Retrieved Chunks (Top-K)</div></div>'
             f'<div class="db">{cards}</div><div class="df"><div class="r"><span>Search Mode:</span>'
             f'<span class="mono">{mode}</span></div><div class="q">"{esc(rq)}"</div></div></div>')


def dropzone_html(max_mb: int) -> str:
    return H(f'<div class="dz"><div class="ib">{ic("file-up", 24)}</div><div class="a">Kéo thả tài liệu PDF hoặc DOCX '
             f'vào đây</div><div class="b">Hỗ trợ kích thước tối đa {max_mb}MB/file. Tự động parse OCR và tách đoạn '
             f'văn bản.</div></div>')


def th(text: str, right: bool = False) -> str:
    style = ' style="text-align:right"' if right else ""
    return f'<div class="th mono"{style}>{esc(text)}</div>'


def td(text: str, cls: str = "", style: str = "") -> str:
    return f'<div class="td {cls}" style="{style}">{esc(text)}</div>'


def td_name(d: dict) -> str:
    t = d.get("type")
    color = {"pdf": "var(--rd400)", "docx": "#60a5fa", "url": "var(--am400)"}.get(t, "var(--s400)")
    return f'<div class="td nm2">{ic("globe" if t == "url" else "file-text", 16, color=color)}{esc(d["name"])}</div>'


def td_status(s: str) -> str:
    if s == "ready":
        return '<div class="td"><span class="pill ok"><span class="dot"></span> Ready / Indexed</span></div>'
    if s == "processing":
        return f'<div class="td"><span class="pill run">{ic("loader-circle", 12, "spin")} Vectorizing...</span></div>'
    if s == "deleting":
        return f'<div class="td"><span class="pill run">{ic("loader-circle", 12, "spin")} Deleting vectors...</span></div>'
    return '<div class="td"><span class="pill err">Failed</span></div>'


def card_head(icon: str, text: str) -> str:
    return f'<div class="ch">{ic(icon)}{esc(text)}</div>'


def label(text: str, value: str = "") -> str:
    v = f' <span class="v mono">{esc(value)}</span>' if value else ""
    return f'<div class="lb">{esc(text)}{v}</div>'


TONE_VAL = {"emerald": "#34d399", "indigo": "#818cf8", "slate": "#e2e8f0", "amber": "#fbbf24"}
TONE_DELTA = {"emerald": "#10b981", "indigo": "#6366f1", "slate": "#94a3b8", "amber": "#f59e0b"}
BAR = {"purple": "#a855f7", "blue": "#3b82f6", "amber": "#f59e0b", "emerald": "#10b981"}


def kpis_html(kpis: list[dict]) -> str:
    out = ""
    for k in kpis:
        dl = (f'<span class="dl" style="color:{TONE_DELTA.get(k.get("delta_tone"), "#10b981")}">{esc(k["delta"])}</span>'
              if k.get("delta") else "")
        out += (f'<div class="kpi"><span class="l">{esc(k["label"])}</span><div class="vr"><span class="v mono" '
                f'style="color:{TONE_VAL.get(k.get("tone"), "#e2e8f0")}">{esc(k["value"])}</span>{dl}</div>'
                f'<div class="d">{esc(k["desc"])}</div></div>')
    return H(f'<div class="kpis">{out}</div>')


def latency_html(items: list[dict]) -> str:
    rows = "".join(
        f'<div class="it"><div class="t"><span>{esc(i["label"])}</span><span>{esc(i["text"])}</span></div>'
        f'<div class="bar"><i style="width:{int(i["pct"])}%;background:{BAR.get(i.get("color"), "#6366f1")}"></i></div></div>'
        for i in items)
    return H(f'<div class="lat"><div class="h">Phân bổ Thời gian Xử lý Trả lời (Total Latency Breakdown)</div>'
             f'<div class="its mono">{rows}</div></div>')


def rerank_html() -> str:
    return H('<div class="rrt"><div class="a">Re-ranking Model (Cohere Rerank v3)</div><div class="b">Sắp xếp lại thứ tự '
             'ưu tiên của Chunks dựa trên mô hình Re-rank chuyên dụng giúp tăng 35% độ chính xác.</div></div>')


