"""Create an empty multi-user-compatible Qdrant collection for a fresh demo.

The command refuses to overwrite an existing collection.  It creates only the
named dense/sparse schema and payload indexes required by the current append
ingestion flow; it never deletes or changes another collection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import get_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a new empty Vietnamese RAG demo collection.")
    parser.add_argument("--collection", required=True, help="Name of the new, currently non-existent collection.")
    args = parser.parse_args()

    cfg = get_config()
    client = QdrantClient(
        host=cfg.vector_store.host,
        port=cfg.vector_store.port,
        api_key=cfg.vector_store.api_key,
        timeout=cfg.vector_store.timeout,
    )
    try:
        existing = {item.name for item in client.get_collections().collections}
        if args.collection in existing:
            print(f"Collection already exists: {args.collection}", file=sys.stderr)
            return 2

        distance = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclidean": models.Distance.EUCLID,
        }.get(cfg.vector_store.distance.lower(), models.Distance.COSINE)
        client.create_collection(
            collection_name=args.collection,
            vectors_config={
                "dense": models.VectorParams(size=cfg.embedding.dimension, distance=distance),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    index=models.SparseIndexParams(),
                    modifier=models.Modifier.IDF,
                )
            },
        )
        for field_name in ("owner_user_id", "doc_id", "file_name", "section_id", "table_id"):
            client.create_payload_index(
                collection_name=args.collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        client.create_payload_index(
            collection_name=args.collection,
            field_name="chunk_index_in_section",
            field_schema=models.PayloadSchemaType.INTEGER,
            wait=True,
        )
    finally:
        client.close()

    print(f"Created empty collection: {args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
