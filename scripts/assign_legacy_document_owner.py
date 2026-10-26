"""Explicitly assign pre-multi-user documents to one registered account.

This tool is intentionally dry-run by default.  It never deletes, recreates,
or re-embeds a Qdrant collection; with --apply it only writes the ownership
payload to the selected document points and marks the same registry rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdrant_client.http import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.auth import UserStore
from src.ingestion.document_registry import DocumentRegistry
from src.ingestion.indexer import QdrantIndexer


def main() -> int:
    parser = argparse.ArgumentParser(description="Assign selected legacy documents to a registered user.")
    parser.add_argument("--owner-email", required=True, help="Email of the user who will own the documents.")
    parser.add_argument("--doc-id", action="append", required=True, help="Legacy document ID; repeat for each document.")
    parser.add_argument("--apply", action="store_true", help="Actually update Qdrant payload and SQLite registry.")
    args = parser.parse_args()

    user = UserStore().get_by_email(args.owner_email)
    if user is None:
        print("User does not exist. Register the account first.", file=sys.stderr)
        return 2

    registry = DocumentRegistry()
    records = []
    for doc_id in args.doc_id:
        record = registry.get(doc_id)
        if record is None:
            print(f"Document not found in registry: {doc_id}", file=sys.stderr)
            return 2
        if record.get("owner_user_id") not in (None, user.user_id):
            print(f"Document {doc_id} already belongs to another user.", file=sys.stderr)
            return 2
        if record.get("status") != "ready" or not record.get("collection_name"):
            print(f"Document {doc_id} is not a ready indexed document.", file=sys.stderr)
            return 2
        records.append(record)

    action = "APPLY" if args.apply else "DRY RUN"
    print(f"{action}: assign {len(records)} document(s) to {user.email} ({user.user_id}).")
    for record in records:
        print(f"- {record['doc_id']} | {record['file_name']} | {record['collection_name']}")

    if not args.apply:
        print("No data changed. Re-run with --apply after verifying the list.")
        return 0

    for record in records:
        indexer = QdrantIndexer(collection_name=record["collection_name"])
        indexer.ensure_owner_payload_index(record["collection_name"])
        point_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=record["doc_id"]),
                )
            ]
        )
        indexer.client.set_payload(
            collection_name=record["collection_name"],
            payload={"owner_user_id": user.user_id},
            points=models.FilterSelector(filter=point_filter),
            wait=True,
        )
        registry.assign_owner(record["doc_id"], user.user_id)
    print("Ownership payload and registry records updated. No embeddings were recalculated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
