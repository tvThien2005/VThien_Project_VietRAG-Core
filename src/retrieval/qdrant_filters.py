from typing import Optional

from qdrant_client.http import models


def build_payload_filter(
    *,
    doc_id: Optional[str] = None,
    file_name: Optional[str] = None,
    section_id: Optional[str] = None,
    table_id: Optional[str] = None,
    chunk_index_min: Optional[int] = None,
    chunk_index_max: Optional[int] = None,
) -> Optional[models.Filter]:
    conditions = []
    for field_name, value in (
        ("doc_id", doc_id),
        ("file_name", file_name),
        ("section_id", section_id),
        ("table_id", table_id),
    ):
        if value is not None:
            conditions.append(
                models.FieldCondition(
                    key=field_name,
                    match=models.MatchValue(value=value),
                )
            )

    if chunk_index_min is not None or chunk_index_max is not None:
        conditions.append(
            models.FieldCondition(
                key="chunk_index_in_section",
                range=models.Range(gte=chunk_index_min, lte=chunk_index_max),
            )
        )

    return models.Filter(must=conditions) if conditions else None
