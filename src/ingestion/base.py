from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from src.schemas.document import Chunk, Document


class BaseIngestionPipeline(ABC):
    """Giao diện chuẩn cho pipeline nạp và xử lý tài liệu."""

    @abstractmethod
    def run(self, source_path: Union[str, Path]) -> List[Chunk]:
        """Thực thi toàn bộ luồng đọc, làm sạch và chia nhỏ tài liệu."""
        pass
