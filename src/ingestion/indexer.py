from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import get_config
from src.ingestion.embedder import BGEEmbedder
from src.logger import get_logger
from src.schemas.document import Chunk

logger = get_logger("indexer")


class ChunkEmbedder(Protocol):
    model_name: str
    dimension: int

    def embed_chunks(
        self,
        chunks: List[Chunk],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> List[List[float]]:
        ...


class QdrantIndexer:
    """Manage Qdrant collections and index document chunks."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        bm25_vector_name: str = "bm25",
        bm25_model: str = "Qdrant/bm25",
    ):
        self.cfg = get_config()
        self.host = host or self.cfg.vector_store.host
        self.port = port or self.cfg.vector_store.port
        self.collection_name = collection_name or self.cfg.vector_store.collection_name
        self.api_key = api_key or self.cfg.vector_store.api_key
        self.timeout = timeout or self.cfg.vector_store.timeout
        self.bm25_vector_name = bm25_vector_name
        self.bm25_model = bm25_model

        logger.info("Connecting to Qdrant at http://%s:%s", self.host, self.port)
        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def is_connected(self) -> bool:
        """Check whether Qdrant is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.warning(
                "Cannot connect to Qdrant at http://%s:%s: %s",
                self.host,
                self.port,
                exc,
            )
            return False

    @staticmethod
    def _validate_chunk_version(chunks: Sequence[Chunk]) -> tuple[str, str, int]:
        chunk_versions = {chunk.metadata.get("chunking_version") for chunk in chunks}
        tokenizers = {chunk.metadata.get("tokenizer") for chunk in chunks}
        token_limits = {chunk.metadata.get("token_limit") for chunk in chunks}
        if None in chunk_versions or len(chunk_versions) != 1:
            raise ValueError("Versioned indexing requires one explicit chunking_version.")
        if None in tokenizers or len(tokenizers) != 1:
            raise ValueError("Versioned indexing requires one explicit chunk tokenizer.")
        if None in token_limits or len(token_limits) != 1:
            raise ValueError("Versioned indexing requires one explicit chunk token limit.")
        return (
            str(next(iter(chunk_versions))),
            str(next(iter(tokenizers))),
            int(next(iter(token_limits))),
        )

    @classmethod
    def versioned_collection_name(
        cls,
        chunks: Sequence[Chunk],
        embedder: ChunkEmbedder,
        base_name: Optional[str] = None,
    ) -> str:
        chunk_version, tokenizer, token_limit = cls._validate_chunk_version(chunks)
        signature = "|".join(
            (
                chunk_version,
                tokenizer,
                str(token_limit),
                embedder.model_name,
                str(embedder.dimension),
                str(getattr(embedder, "normalize_embeddings", None)),
                get_config().vector_store.distance,
                "Qdrant/bm25",
            )
        )
        version = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]
        base = base_name or get_config().vector_store.collection_name
        base_slug = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{base_slug}_{version}_{timestamp}"

    def create_collection_if_not_exists(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        distance: str = "Cosine",
        recreate: bool = False,
    ) -> bool:
        """Create a legacy unnamed-dense collection, optionally recreating it."""
        c_name = collection_name or self.collection_name
        dim = dimension or self.cfg.embedding.dimension
        distance_map = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclidean": models.Distance.EUCLID,
        }
        dist = distance_map.get(distance.lower(), models.Distance.COSINE)
        collections = [item.name for item in self.client.get_collections().collections]

        if c_name in collections:
            if recreate:
                logger.warning("Deleting existing collection '%s' as explicitly requested.", c_name)
                self.client.delete_collection(collection_name=c_name)
            else:
                logger.info("Collection '%s' already exists.", c_name)
                return True

        self.client.create_collection(
            collection_name=c_name,
            vectors_config=models.VectorParams(size=dim, distance=dist),
        )
        logger.info("Created collection '%s' (dimension=%s).", c_name, dim)
        return True

    def index_chunks(
        self,
        chunks: List[Chunk],
        embedder: Optional[ChunkEmbedder] = None,
        batch_size: int = 64,
        recreate_collection: bool = False,
    ) -> int:
        """Index dense vectors into the configured legacy collection."""
        if not chunks:
            logger.warning("No chunks to index.")
            return 0
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if not self.is_connected():
            raise ConnectionError(
                f"Cannot connect to Qdrant at http://{self.host}:{self.port}. "
                "Check that the Qdrant service is running."
            )

        embedder = embedder or BGEEmbedder()
        self.create_collection_if_not_exists(
            dimension=embedder.dimension,
            recreate=recreate_collection,
        )
        embeddings = embedder.embed_chunks(chunks, show_progress_bar=True)
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedder returned {len(embeddings)} vectors for {len(chunks)} chunks."
            )

        indexed_count = 0
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = embeddings[start : start + batch_size]
            points = []
            for chunk, vector in zip(batch_chunks, batch_vectors):
                payload = {
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "char_count": chunk.char_count,
                    "word_count": chunk.word_count,
                    **chunk.metadata,
                }
                try:
                    point_id = str(uuid.UUID(chunk.chunk_id))
                except ValueError:
                    point_id = str(uuid.uuid4())
                points.append(
                    models.PointStruct(id=point_id, vector=vector, payload=payload)
                )
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            indexed_count += len(points)
            logger.info("Indexed %s/%s legacy points.", indexed_count, len(chunks))
        return indexed_count

    def index_chunks_versioned(
        self,
        chunks: List[Chunk],
        embedder: Optional[ChunkEmbedder] = None,
        batch_size: int = 64,
        collection_name: Optional[str] = None,
    ) -> int:
        """Build and verify a new isolated collection; never replace existing data."""
        if not chunks:
            logger.warning("No chunks to index.")
            return 0
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if not self.is_connected():
            raise ConnectionError(
                f"Cannot connect to Qdrant at http://{self.host}:{self.port}."
            )

        embedder = embedder or BGEEmbedder()
        chunk_version, chunk_tokenizer, token_limit = self._validate_chunk_version(chunks)
        target_collection = collection_name or self.versioned_collection_name(
            chunks,
            embedder,
        )
        existing = {
            item.name for item in self.client.get_collections().collections
        }
        if target_collection in existing:
            raise ValueError(
                f"Refusing to overwrite existing collection '{target_collection}'. "
                "Choose a new collection name; existing data was not modified."
            )

        point_ids = []
        for chunk in chunks:
            try:
                point_ids.append(str(uuid.UUID(chunk.chunk_id)))
            except (ValueError, AttributeError) as exc:
                raise ValueError(
                    f"Versioned indexing requires UUID chunk IDs; got {chunk.chunk_id!r}."
                ) from exc
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("Chunk IDs must be unique within a versioned index.")

        embeddings = embedder.embed_chunks(chunks, show_progress_bar=True)
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedder returned {len(embeddings)} vectors for {len(chunks)} chunks."
            )
        if any(len(vector) != embedder.dimension for vector in embeddings):
            raise ValueError("Embedding dimension does not match the configured embedder.")

        distance_map = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclidean": models.Distance.EUCLID,
        }
        distance = distance_map.get(
            self.cfg.vector_store.distance.lower(),
            models.Distance.COSINE,
        )
        self.client.create_collection(
            collection_name=target_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=embedder.dimension,
                    distance=distance,
                )
            },
            sparse_vectors_config={
                self.bm25_vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(),
                    modifier=models.Modifier.IDF,
                )
            },
        )
        logger.info(
            "Created isolated versioned collection '%s'; existing collections remain untouched.",
            target_collection,
        )
        for field_name in ("owner_user_id", "doc_id", "file_name", "section_id", "table_id"):
            self.client.create_payload_index(
                collection_name=target_collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        self.client.create_payload_index(
            collection_name=target_collection,
            field_name="chunk_index_in_section",
            field_schema=models.PayloadSchemaType.INTEGER,
            wait=True,
        )

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = embeddings[start : start + batch_size]
            points: List[models.PointStruct] = []
            for chunk, vector, point_id in zip(
                batch_chunks,
                batch_vectors,
                point_ids[start : start + batch_size],
            ):
                source_table = chunk.metadata.get("source_table") or {}
                payload = {
                    **chunk.metadata,
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "file_name": chunk.metadata.get("file_name", ""),
                    "chunking_version": chunk_version,
                    "embedding_model": embedder.model_name,
                    "embedding_dimension": embedder.dimension,
                    "embedding_normalized": getattr(
                        embedder,
                        "normalize_embeddings",
                        None,
                    ),
                    "sparse_model": self.bm25_model,
                    "index_collection": target_collection,
                    "table_id": chunk.metadata.get("table_id")
                    or source_table.get("table_id"),
                }
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "dense": vector,
                            self.bm25_vector_name: models.Document(
                                text=chunk.text,
                                model=self.bm25_model,
                            ),
                        },
                        payload=payload,
                    )
                )
            self.client.upsert(
                collection_name=target_collection,
                points=points,
                wait=True,
            )
            logger.info(
                "Indexed %s/%s points into '%s'.",
                min(start + len(points), len(chunks)),
                len(chunks),
                target_collection,
            )

        self._verify_versioned_collection(
            target_collection,
            point_ids,
            embedder,
            embeddings[0],
            chunk_version,
            chunk_tokenizer,
            token_limit,
        )
        logger.info(
            "Verified versioned index '%s': %s points, dense=%s, sparse=%s.",
            target_collection,
            len(point_ids),
            embedder.dimension,
            self.bm25_vector_name,
        )
        return len(point_ids)

    def _document_points(
        self,
        collection_name: str,
        doc_id: str,
        owner_user_id: Optional[str] = None,
    ) -> List[Any]:
        """Return every point belonging to one document without loading vectors."""
        points: List[Any] = []
        offset = None
        conditions = [
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            )
        ]
        if owner_user_id is not None:
            conditions.insert(
                0,
                models.FieldCondition(
                    key="owner_user_id",
                    match=models.MatchValue(value=owner_user_id),
                ),
            )
        query_filter = models.Filter(must=conditions)
        while True:
            batch, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if offset is None:
                return points

    def ensure_owner_payload_index(self, collection_name: Optional[str] = None) -> None:
        """Create the ownership filter index without recreating or rewriting points."""
        self.client.create_payload_index(
            collection_name=collection_name or self.collection_name,
            field_name="owner_user_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def _validate_append_schema(
        self,
        collection_name: str,
        embedder: ChunkEmbedder,
        chunk_version: str,
        tokenizer: str,
        token_limit: int,
    ) -> None:
        """Refuse an append when the existing versioned index is incompatible."""
        collections = {item.name for item in self.client.get_collections().collections}
        if collection_name not in collections:
            raise ValueError(
                f"Collection '{collection_name}' does not exist; append never creates a collection."
            )

        info = self.client.get_collection(collection_name=collection_name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or set(vectors) != {"dense"}:
            raise ValueError(
                f"Collection '{collection_name}' is not a versioned named-dense collection."
            )
        if vectors["dense"].size != embedder.dimension:
            raise ValueError(
                f"Collection '{collection_name}' has dimension {vectors['dense'].size}, "
                f"but the configured embedder returns {embedder.dimension}."
            )

        sparse_vectors = getattr(info.config.params, "sparse_vectors", None) or {}
        if self.bm25_vector_name not in sparse_vectors:
            raise ValueError(
                f"Collection '{collection_name}' is missing sparse vector '{self.bm25_vector_name}'."
            )

        sample, _ = self.client.scroll(
            collection_name=collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not sample:
            return
        payload = sample[0].payload or {}
        expected = {
            "embedding_model": embedder.model_name,
            "embedding_dimension": embedder.dimension,
            "chunking_version": chunk_version,
            "tokenizer": tokenizer,
            "token_limit": token_limit,
            "index_collection": collection_name,
        }
        for field, value in expected.items():
            if field not in payload:
                raise ValueError(
                    f"Collection '{collection_name}' lacks version metadata '{field}'; refusing unsafe append."
                )
            if payload[field] != value:
                raise ValueError(
                    f"Collection '{collection_name}' has incompatible {field}: "
                    f"{payload[field]!r} != {value!r}."
                )

    def append_chunks_versioned(
        self,
        chunks: List[Chunk],
        embedder: Optional[ChunkEmbedder] = None,
        batch_size: int = 64,
        collection_name: Optional[str] = None,
        replace_document: bool = False,
    ) -> int:
        """Append one document to an existing versioned collection safely.

        Existing collections are never recreated.  A duplicate ``doc_id`` is
        rejected by default.  With ``replace_document=True``, new points are
        upserted and verified before stale points of that exact document are
        removed; unrelated Qdrant points are never touched.
        """
        if not chunks:
            raise ValueError("No chunks to append.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if not self.is_connected():
            raise ConnectionError(
                f"Cannot connect to Qdrant at http://{self.host}:{self.port}."
            )

        document_ids = {chunk.doc_id for chunk in chunks}
        if len(document_ids) != 1 or not next(iter(document_ids)):
            raise ValueError("Append requires chunks from exactly one non-empty doc_id.")
        doc_id = next(iter(document_ids))
        embedder = embedder or BGEEmbedder()
        target_collection = collection_name or self.collection_name
        chunk_version, tokenizer, token_limit = self._validate_chunk_version(chunks)
        self._validate_append_schema(
            target_collection,
            embedder,
            chunk_version,
            tokenizer,
            token_limit,
        )
        # This is idempotent in Qdrant and only builds an index.  It never
        # recreates the active collection or mutates existing point payloads.
        self.ensure_owner_payload_index(target_collection)

        existing_points = self._document_points(target_collection, doc_id)
        if existing_points and not replace_document:
            raise FileExistsError(
                f"Document '{doc_id}' already has {len(existing_points)} points in '{target_collection}'."
            )

        point_ids: List[str] = []
        for chunk in chunks:
            try:
                point_ids.append(str(uuid.UUID(chunk.chunk_id)))
            except (ValueError, AttributeError) as exc:
                raise ValueError(
                    f"Versioned append requires UUID chunk IDs; got {chunk.chunk_id!r}."
                ) from exc
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("Chunk IDs must be unique for an append.")

        embeddings = embedder.embed_chunks(chunks, show_progress_bar=False)
        if len(embeddings) != len(chunks) or any(
            len(vector) != embedder.dimension for vector in embeddings
        ):
            raise ValueError("Embedding output does not match the configured vector dimension.")

        for start in range(0, len(chunks), batch_size):
            points: List[models.PointStruct] = []
            for chunk, vector, point_id in zip(
                chunks[start : start + batch_size],
                embeddings[start : start + batch_size],
                point_ids[start : start + batch_size],
            ):
                source_table = chunk.metadata.get("source_table") or {}
                payload = {
                    **chunk.metadata,
                    "text": chunk.text,
                    "doc_id": doc_id,
                    "owner_user_id": chunk.metadata.get("owner_user_id"),
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "file_name": chunk.metadata.get("file_name", ""),
                    "chunking_version": chunk_version,
                    "embedding_model": embedder.model_name,
                    "embedding_dimension": embedder.dimension,
                    "embedding_normalized": getattr(embedder, "normalize_embeddings", None),
                    "sparse_model": self.bm25_model,
                    "index_collection": target_collection,
                    "table_id": chunk.metadata.get("table_id") or source_table.get("table_id"),
                }
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "dense": vector,
                            self.bm25_vector_name: models.Document(
                                text=chunk.text,
                                model=self.bm25_model,
                            ),
                        },
                        payload=payload,
                    )
                )
            self.client.upsert(collection_name=target_collection, points=points, wait=True)

        indexed_points = self._document_points(target_collection, doc_id)
        indexed_ids = {str(point.id) for point in indexed_points}
        expected_ids = set(point_ids)
        if not expected_ids.issubset(indexed_ids):
            raise RuntimeError("Qdrant verification failed after append; new points are missing.")

        stale_ids = {str(point.id) for point in existing_points} - expected_ids
        if stale_ids:
            self.client.delete(
                collection_name=target_collection,
                points_selector=models.PointIdsList(points=list(stale_ids)),
                wait=True,
            )
        final_points = self._document_points(target_collection, doc_id)
        if {str(point.id) for point in final_points} != expected_ids:
            raise RuntimeError("Qdrant verification failed after document replacement.")
        logger.info(
            "Appended %s chunks for doc_id=%s into '%s'.",
            len(chunks), doc_id, target_collection,
        )
        return len(chunks)

    def _verify_versioned_collection(
        self,
        collection_name: str,
        expected_ids: Sequence[str],
        embedder: ChunkEmbedder,
        sample_embedding: List[float],
        chunk_version: str,
        chunk_tokenizer: str,
        token_limit: int,
    ) -> None:
        info = self.client.get_collection(collection_name=collection_name)
        if info.points_count != len(expected_ids):
            raise RuntimeError(
                f"Collection '{collection_name}' has {info.points_count} points; "
                f"expected {len(expected_ids)}."
            )

        configured_vectors = info.config.params.vectors
        if not isinstance(configured_vectors, dict) or set(configured_vectors) != {"dense"}:
            raise RuntimeError(
                f"Collection '{collection_name}' does not expose the expected named dense vector."
            )
        if configured_vectors["dense"].size != embedder.dimension:
            raise RuntimeError(
                f"Collection '{collection_name}' dense dimension does not match the embedder."
            )
        configured_sparse = info.config.params.sparse_vectors or {}
        if set(configured_sparse) != {self.bm25_vector_name}:
            raise RuntimeError(
                f"Collection '{collection_name}' sparse vectors do not match "
                f"the expected vector '{self.bm25_vector_name}'."
            )

        seen_ids = set()
        payload_count = 0
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                seen_ids.add(str(point.id))
                payload = point.payload or {}
                required_fields = (
                    "text",
                    "doc_id",
                    "file_name",
                    "source",
                    "chunk_id",
                    "source_block_ids",
                    "block_types",
                    "chunk_has_table",
                    "chunking_version",
                )
                if not all(field in payload for field in required_fields) or not all(
                    payload.get(field)
                    for field in (
                        "text",
                        "doc_id",
                        "file_name",
                        "source",
                        "chunk_id",
                        "source_block_ids",
                        "block_types",
                    )
                ):
                    raise RuntimeError(
                        f"Point '{point.id}' is missing required retrieval/source payload fields."
                    )
                if str(payload["chunk_id"]) != str(point.id):
                    raise RuntimeError(
                        f"Point '{point.id}' has a mismatched source chunk ID in its payload."
                    )
                if (
                    payload.get("chunking_version") != chunk_version
                    or payload.get("tokenizer") != chunk_tokenizer
                    or payload.get("token_limit") != token_limit
                    or payload.get("index_collection") != collection_name
                ):
                    raise RuntimeError(
                        f"Point '{point.id}' contains incompatible chunk/index version metadata."
                    )
                if payload.get("embedding_model") != embedder.model_name:
                    raise RuntimeError(
                        f"Point '{point.id}' has incompatible embedding model metadata."
                    )
                if payload.get("embedding_dimension") != embedder.dimension:
                    raise RuntimeError(
                        f"Point '{point.id}' has an incompatible embedding dimension."
                    )
                if payload.get("sparse_model") != self.bm25_model:
                    raise RuntimeError(
                        f"Point '{point.id}' has an incompatible sparse model marker."
                    )
                if payload.get("chunk_has_table") and not payload.get("table_id"):
                    raise RuntimeError(
                        f"Table point '{point.id}' is missing its table_id."
                    )
                payload_count += 1
            if offset is None:
                break
        if seen_ids != set(expected_ids) or payload_count != len(expected_ids):
            raise RuntimeError(
                f"Collection '{collection_name}' point IDs or payload count do not match input chunks."
            )

        dense_probe = self.client.query_points(
            collection_name=collection_name,
            query=sample_embedding,
            using="dense",
            limit=1,
            with_payload=True,
        )
        sparse_probe = self.client.query_points(
            collection_name=collection_name,
            query=models.Document(text="kiểm tra chỉ mục", model=self.bm25_model),
            using=self.bm25_vector_name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if (
            not dense_probe.points
            or not sparse_probe.points
            or str(dense_probe.points[0].id) not in set(expected_ids)
            or str(sparse_probe.points[0].id) not in set(expected_ids)
        ):
            raise RuntimeError(
                f"Collection '{collection_name}' failed dense or sparse query verification."
            )

    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Return point count and dense/sparse vector configuration."""
        c_name = collection_name or self.collection_name
        try:
            info = self.client.get_collection(collection_name=c_name)
            configured_vectors = info.config.params.vectors
            dense_vectors = (
                configured_vectors
                if isinstance(configured_vectors, dict)
                else {"": configured_vectors}
            )
            sparse_vectors = info.config.params.sparse_vectors or {}
            vector_sizes = {
                name: vector.size for name, vector in dense_vectors.items()
            }
            return {
                "name": c_name,
                "status": str(info.status),
                "vectors_count": info.points_count or info.indexed_vectors_count,
                "vector_size": vector_sizes.get("dense")
                or next(iter(vector_sizes.values()), None),
                "vector_sizes": vector_sizes,
                "vector_names": list(dense_vectors),
                "sparse_vector_names": list(sparse_vectors),
                "distance": str(next(iter(dense_vectors.values())).distance),
            }
        except Exception as exc:
            logger.error("Failed to read collection '%s': %s", c_name, exc)
            raise
