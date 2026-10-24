from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from src.schemas.blocks import ParsedBlock
from src.schemas.document import Document


class BaseLoader(ABC):
    """Interface cơ sở cho tất cả các Document Loader."""

    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """Đọc và trích xuất danh sách Document từ đường dẫn file."""
        pass

    def load_blocks(self, file_path: Union[str, Path]) -> List[ParsedBlock]:
        """Return normalized parsing blocks when the loader supports them.

        ``load`` remains the compatibility API for the existing ingestion
        pipeline.  Loaders that have not migrated fail explicitly instead of
        silently inventing block locations or table structure.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose normalized parsed blocks yet."
        )
