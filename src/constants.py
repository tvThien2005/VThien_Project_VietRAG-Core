"""Các hằng số toàn cục cho hệ thống Vietnamese RAG."""

# Extension file hỗ trợ
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# Bộ dấu phân tách chunk chuẩn cho tiếng Việt
VIETNAMESE_SEPARATORS = [
    "\n\n",   # Đoạn văn
    "\n",     # Dòng mới
    ". ",     # Hết câu chấm
    "? ",     # Hết câu hỏi
    "! ",     # Hết câu cảm thán
    "; ",     # Chấm phẩy
    ", ",     # Dấu phẩy
    " ",      # Khoảng trắng từ
    "",       # Ký tự
]

# Metadata keys
META_DOC_ID = "doc_id"
META_SOURCE = "source"
META_FILE_NAME = "file_name"
META_FILE_TYPE = "file_type"
META_PAGE_NUMBER = "page_number"
META_TOTAL_PAGES = "total_pages"
META_CHUNK_INDEX = "chunk_index"
META_CHAR_COUNT = "char_count"
META_WORD_COUNT = "word_count"
