"""Background ingestion service for files uploaded through the API."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from src.config import PROJECT_ROOT
from src.ingestion.document_registry import DocumentRegistry
from src.ingestion.embedder import BGEEmbedder
from src.ingestion.indexer import QdrantIndexer
from src.ingestion.pipeline import IngestionPipeline
from src.schemas.document import Chunk
from src.utils.io import save_jsonl


def document_id_for_bytes(content: bytes) -> str:
    """Content-addressed identity makes duplicate-upload handling deterministic."""
    return hashlib.sha256(content).hexdigest()


def chunk_artifact_path(doc_id: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / "documents" / doc_id / "chunks.jsonl"


def serialize_chunks(chunks: Iterable[Chunk]) -> list[dict]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "char_count": chunk.char_count,
            "word_count": chunk.word_count,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]


def process_document_job(
    *,
    registry: DocumentRegistry,
    doc_id: str,
    source_path: str | Path,
    file_name: str,
    owner_user_id: str,
    replace_existing: bool = False,
) -> None:
    """Run outside the request handler; all errors become durable job state."""
    registry.mark_processing(doc_id)
    try:
        pipeline = IngestionPipeline()
        chunks = pipeline.run(source_path=source_path, save_to_processed=False)
        if not chunks:
            raise ValueError("Không trích xuất được chunk nào từ tài liệu.")

        for chunk in chunks:
            chunk.doc_id = doc_id
            chunk.metadata = {
                **chunk.metadata,
                "doc_id": doc_id,
                "file_name": file_name,
                "owner_user_id": owner_user_id,
            }

        artifact = chunk_artifact_path(doc_id)
        save_jsonl(serialize_chunks(chunks), artifact)

        indexer = QdrantIndexer()
        embedder = BGEEmbedder()
        indexed_count = indexer.append_chunks_versioned(
            chunks,
            embedder=embedder,
            replace_document=replace_existing,
        )
        registry.mark_ready(
            doc_id,
            chunk_count=indexed_count,
            embedding_model=embedder.model_name,
            collection_name=indexer.collection_name,
            artifact_path=str(artifact),
        )
    except Exception as exc:
        registry.mark_failed(doc_id, str(exc))
