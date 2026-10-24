from pathlib import Path
from typing import List, Union

from src.ingestion.loaders.base import BaseLoader
from src.logger import get_logger
from src.schemas.document import Document
from src.utils.text import clean_text

logger = get_logger("txt_loader")


class TxtLoader(BaseLoader):
    """Loader chuyên xử lý đọc các file văn bản thuần (.txt, .md)."""

    def __init__(self, clean_content: bool = True, encoding: str = "utf-8"):
        self.clean_content = clean_content
        self.encoding = encoding

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

        logger.info(f"Đang đọc file text: {path.name}")
        try:
            with open(path, "r", encoding=self.encoding) as f:
                raw_text = f.read()

            text = clean_text(raw_text) if self.clean_content else raw_text.strip()
            if not text:
                return []

            metadata = {
                "source": str(path),
                "file_name": path.name,
                "file_type": path.suffix.lstrip("."),
            }

            return [Document(text=text, metadata=metadata)]
        except Exception as e:
            logger.error(f"Lỗi khi đọc file {path.name}: {e}")
            raise
