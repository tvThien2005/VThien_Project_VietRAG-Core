from pathlib import Path
from typing import Any, List, Optional, Union

from src.config import get_config
from src.ingestion.base import BaseIngestionPipeline
from src.ingestion.chunkers.recursive_chunker import RecursiveCharacterChunker
from src.ingestion.chunkers.structured_chunker import StructuredChunker
from src.ingestion.loaders import get_loader
from src.logger import get_logger
from src.schemas.blocks import ParsedBlock
from src.schemas.document import Chunk, Document
from src.utils.io import save_jsonl

logger = get_logger("ingestion_pipeline")


class IngestionPipeline(BaseIngestionPipeline):
    """Pipeline nạp, tiền xử lý và cắt nhỏ tài liệu."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunker: Optional[RecursiveCharacterChunker] = None,
        structured_chunker: Optional[StructuredChunker] = None,
    ):
        self.cfg = get_config()
        self.chunker = chunker or RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.structured_chunker = structured_chunker or StructuredChunker(
            max_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
        )

    def load_blocks(self, file_path: Union[str, Path]) -> List[ParsedBlock]:
        """Load normalized blocks from a loader with structured output."""
        path = Path(file_path)
        return get_loader(path).load_blocks(path)

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Đọc và trích xuất tài liệu từ một file duy nhất."""
        path = Path(file_path)
        loader = get_loader(path)
        return loader.load(path)

    def load_directory(self, dir_path: Union[str, Path]) -> List[Document]:
        """Quét và đọc tất cả các file hợp lệ trong thư mục."""
        path = Path(dir_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Thư mục không tồn tại: {path}")

        supported_exts = set(self.cfg.data.supported_extensions)
        all_documents: List[Document] = []

        logger.info(f"Bắt đầu quét tài liệu trong thư mục: {path}")
        for file in sorted(path.rglob("*")):
            if file.is_file() and file.suffix.lower() in supported_exts:
                try:
                    docs = self.load_file(file)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý file {file.name}: {e}")

        logger.info(f"Đã nạp tổng cộng {len(all_documents)} Document từ {path}")
        return all_documents

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Cắt nhỏ danh sách các Document thành các Chunks."""
        return self.chunker.split_documents(documents)

    def chunk_blocks(self, blocks: List[ParsedBlock]) -> List[Chunk]:
        """Chunk parser blocks without flattening their section/table structure."""
        return self.structured_chunker.split_blocks(blocks)

    def _chunk_file(self, file_path: Path) -> List[Chunk]:
        loader = get_loader(file_path)
        try:
            blocks = loader.load_blocks(file_path)
        except NotImplementedError:
            return self.chunker.split_documents(loader.load(file_path))
        return self.structured_chunker.split_blocks(blocks)

    def run(
        self,
        source_path: Optional[Union[str, Path]] = None,
        save_to_processed: bool = True,
    ) -> List[Chunk]:
        """Chạy toàn bộ pipeline từ đọc file đến cắt chunks.
        
        Nếu source_path là thư mục, nạp toàn bộ file trong thư mục.
        Nếu source_path là file, chỉ nạp file đó.
        Nếu không truyền source_path, mặc định nạp từ thư mục data/raw.
        """
        target = Path(source_path) if source_path else self.cfg.data.raw_path
        if not target.exists():
            raise FileNotFoundError(f"Nguồn tài liệu không tồn tại: {target}")

        supported_exts = set(self.cfg.data.supported_extensions)
        source_files = (
            [
                path
                for path in sorted(target.rglob("*"))
                if path.is_file() and path.suffix.lower() in supported_exts
            ]
            if target.is_dir()
            else [target]
        )
        chunks: List[Chunk] = []
        for source_file in source_files:
            try:
                chunks.extend(self._chunk_file(source_file))
            except Exception as exc:
                if not target.is_dir():
                    raise
                logger.error(f"Lỗi khi xử lý file {source_file.name}: {exc}")

        if not chunks:
            logger.warning(f"Không tìm thấy tài liệu nào tại: {target}")
            return []

        # Lưu các chunk đã xử lý ra JSONL nếu được yêu cầu
        if save_to_processed and chunks:
            output_file = self.cfg.data.processed_path / "chunks.jsonl"
            data_to_save = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "chunk_index": c.chunk_index,
                    "char_count": c.char_count,
                    "word_count": c.word_count,
                    "metadata": c.metadata,
                }
                for c in chunks
            ]
            save_jsonl(data_to_save, output_file)
            logger.info(f"Đã lưu {len(chunks)} chunks vào file: {output_file}")

        return chunks

    def to_llama_nodes(self, chunks: List[Chunk]) -> List[Any]:
        """Chuyển danh sách Chunk thành danh sách LlamaIndex TextNode."""
        return [c.to_llama_node() for c in chunks]
