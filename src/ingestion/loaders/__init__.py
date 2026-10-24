from pathlib import Path
from typing import Union

from src.ingestion.loaders.base import BaseLoader
from src.ingestion.loaders.docx_loader import DocxLoader
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.loaders.txt_loader import TxtLoader


def get_loader(file_path: Union[str, Path], clean_content: bool = True) -> BaseLoader:
    """Trả về Loader tương ứng dựa vào phần mở rộng của file."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return PDFLoader(clean_content=clean_content)
    elif ext in [".docx", ".doc"]:
        return DocxLoader(clean_content=clean_content)
    elif ext in [".txt", ".md"]:
        return TxtLoader(clean_content=clean_content)
    else:
        raise ValueError(f"Định dạng file '{ext}' chưa được hỗ trợ.")


__all__ = ["BaseLoader", "PDFLoader", "DocxLoader", "TxtLoader", "get_loader"]
