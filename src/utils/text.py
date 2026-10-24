import re
import unicodedata
from typing import Optional


def normalize_vietnamese_unicode(text: str) -> str:
    """Chuẩn hóa bảng mã Unicode Tiếng Việt sang dạng dựng sẵn (NFC).
    Giúp tránh tình trạng lỗi embedding/tìm kiếm do văn bản bị phân rã (NFD).
    """
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def clean_text(text: str, remove_extra_whitespace: bool = True) -> str:
    """Làm sạch văn bản thô trích xuất từ PDF hoặc Word:
    
    - Chuẩn hóa Unicode NFC.
    - Loại bỏ ký tự điều khiển ẩn (ngoại trừ \\n, \\t).
    - Nối các từ bị gãy dòng bằng dấu gạch ngang (ví dụ: 'thông-\\ntin' -> 'thông tin').
    - Chuẩn hóa khoảng trắng và dòng trống liên tiếp.
    """
    if not text:
        return ""

    # 1. Chuẩn hóa NFC
    text = normalize_vietnamese_unicode(text)

    # 2. Thay thế ký tự ngắt dòng Windows (\r\n) thành \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Nối các từ bị đứt đoạn gạch nối ở cuối dòng (vd: 'phát-\\ntriển' -> 'phát triển')
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # 4. Loại bỏ ký tự điều khiển lạ (non-printable), giữ lại dấu tiếng Việt và \n \t
    text = "".join(ch for ch in text if ch.isprintable() or ch in ("\n", "\t"))

    if remove_extra_whitespace:
        # Thay thế nhiều khoảng trắng ngang (spaces, tabs) thành 1 khoảng trắng
        text = re.sub(r"[ \t]+", " ", text)
        # Giới hạn tối đa 2 dấu xuống dòng liên tiếp
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

    return text


def count_words(text: str) -> int:
    """Đếm số lượng từ trong đoạn văn."""
    if not text:
        return 0
    return len(text.split())
