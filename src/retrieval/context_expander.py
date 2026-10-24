from typing import Any, Dict, List, Optional, Sequence, Tuple

from qdrant_client.http import models
from transformers import AutoTokenizer

from src.config import ContextExpansionConfig, get_config
from src.generation.prompt_builder import PromptBuilder
from src.logger import get_logger
from src.retrieval.qdrant_filters import build_payload_filter
from src.schemas.blocks import TableData
from src.schemas.response import RetrievedChunk

logger = get_logger("context_expander")


def create_context_tokenizer() -> Any:
    cfg = get_config()
    expansion_cfg = cfg.retrieval.context_expansion
    if cfg.llm.default_provider == "groq":
        import tiktoken

        model_name = cfg.llm.groq.model_name.rsplit("/", 1)[-1]
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError as exc:
            raise ValueError(
                f"No tiktoken encoding is registered for Groq model {model_name!r}."
            ) from exc

    if cfg.llm.default_provider == "ollama":
        tokenizer_name = expansion_cfg.tokenizer_name or _ollama_tokenizer_name(
            cfg.llm.ollama.model_name
        )
        return AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )

    raise ValueError(
        f"Context tokenization is not configured for LLM provider "
        f"{cfg.llm.default_provider!r}."
    )


def _ollama_tokenizer_name(model_name: str) -> str:
    qwen_match = {
        "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B",
        "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B",
        "qwen2.5:3b": "Qwen/Qwen2.5-3B",
        "qwen2.5:7b": "Qwen/Qwen2.5-7B",
        "qwen2.5:14b": "Qwen/Qwen2.5-14B",
        "qwen2.5:32b": "Qwen/Qwen2.5-32B",
        "qwen2.5:72b": "Qwen/Qwen2.5-72B",
    }
    model_tag = model_name.lower()
    if model_tag in qwen_match:
        return qwen_match[model_tag]
    raise ValueError(
        f"Set retrieval.context_expansion.tokenizer_name for Ollama model "
        f"{model_name!r} so context limits use its actual tokenizer."
    )


class ContextExpander:
    """Add nearby same-section chunks within a complete-prompt token budget."""

    def __init__(
        self,
        *,
        client: Any,
        collection_name: str,
        tokenizer: Any,
        config: Optional[ContextExpansionConfig] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.tokenizer = tokenizer
        self.config = config or get_config().retrieval.context_expansion
        self.prompt_builder = prompt_builder or PromptBuilder()

    def _token_count(self, text: str) -> int:
        if hasattr(self.tokenizer, "encode"):
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                try:
                    return len(self.tokenizer.encode(text, disallowed_special=()))
                except TypeError:
                    return len(self.tokenizer.encode(text))
        raise TypeError("The context tokenizer must provide encode().")

    def _prompt_fits(self, query: str, chunks: Sequence[RetrievedChunk]) -> bool:
        system_prompt, user_prompt = self.prompt_builder.build_prompt(query, list(chunks))
        return (
            self._token_count(system_prompt) + self._token_count(user_prompt)
            <= self.config.max_prompt_tokens
        )

    def _scroll_chunks(
        self,
        query_filter: models.Filter,
        seed: RetrievedChunk,
        *,
        limit: int,
    ) -> List[RetrievedChunk]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        chunks = []
        for point in points:
            payload = dict(point.payload or {})
            text = payload.get("text")
            if not text:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(point.id),
                    doc_id=payload.get("doc_id"),
                    text=text,
                    score=seed.score,
                    rank=seed.rank,
                    metadata={
                        **payload,
                        "is_context_expansion": str(point.id) != seed.chunk_id,
                        "expanded_from_chunk_id": seed.chunk_id,
                    },
                )
            )
        return chunks

    def _neighbors(self, seed: RetrievedChunk) -> List[RetrievedChunk]:
        metadata = seed.metadata
        doc_id = seed.doc_id or metadata.get("doc_id")
        section_id = metadata.get("section_id")
        index = metadata.get("chunk_index_in_section")
        if (
            not doc_id
            or not section_id
            or not isinstance(index, int)
            or self.config.neighbor_window == 0
        ):
            return []

        query_filter = build_payload_filter(
            doc_id=doc_id,
            section_id=section_id,
            chunk_index_min=max(0, index - self.config.neighbor_window),
            chunk_index_max=index + self.config.neighbor_window,
        )
        if query_filter is None:
            return []
        return self._scroll_chunks(
            query_filter,
            seed,
            limit=2 * self.config.neighbor_window + 1,
        )

    @staticmethod
    def _table_text(table: TableData, row_indexes: Sequence[int]) -> str:
        lines = []
        if table.caption:
            lines.append(f"Caption: {table.caption}")

        headers = table.header_rows or ([table.header] if table.header else [])
        for header_index, row in enumerate(headers):
            lines.append("| " + " | ".join(row) + " |")
            if header_index == len(headers) - 1:
                lines.append("| " + " | ".join(["---"] * len(row)) + " |")

        if headers:
            for row_index in row_indexes:
                if row_index < len(table.rows):
                    row = table.rows[row_index]
                    padded = row + [""] * (len(headers[-1]) - len(row))
                    lines.append("| " + " | ".join(padded) + " |")
        else:
            for row_index in row_indexes:
                if row_index < len(table.rows):
                    lines.append(
                        f"Row {row_index + 1}: " + " | ".join(table.rows[row_index])
                    )
        return "\n".join(lines)

    @staticmethod
    def _row_relevance(query: str, row: Sequence[str]) -> int:
        query_terms = {part.casefold() for part in query.split() if len(part) > 1}
        row_text = " ".join(row).casefold()
        return sum(term in row_text for term in query_terms)

    def _fit_table(
        self,
        query: str,
        context: List[RetrievedChunk],
        chunk: RetrievedChunk,
    ) -> Optional[RetrievedChunk]:
        raw_table = chunk.metadata.get("source_table")
        if not isinstance(raw_table, dict):
            return None
        table = TableData.model_validate(raw_table)
        all_indexes = list(range(len(table.rows)))
        ranked_indexes = sorted(
            all_indexes,
            key=lambda row_index: (
                -self._row_relevance(query, table.rows[row_index]),
                row_index,
            ),
        )
        selected: List[int] = []
        base = chunk.model_copy(
            update={
                "text": self._table_text(table, selected),
                "metadata": {
                    **chunk.metadata,
                    "context_table_rows": selected,
                    "context_table_excerpt": True,
                },
            }
        )
        if not self._prompt_fits(query, [*context, base]):
            return None

        for row_index in ranked_indexes:
            candidate_indexes = sorted([*selected, row_index])
            candidate = chunk.model_copy(
                update={
                    "text": self._table_text(table, candidate_indexes),
                    "metadata": {
                        **chunk.metadata,
                        "context_table_rows": candidate_indexes,
                        "context_table_excerpt": True,
                    },
                }
            )
            if self._prompt_fits(query, [*context, candidate]):
                selected = candidate_indexes
                base = candidate
        return base

    @staticmethod
    def _document_order_key(
        chunk: RetrievedChunk,
        document_order: Dict[str, int],
        original_order: Dict[str, int],
    ) -> Tuple[int, int, int, int]:
        metadata = chunk.metadata
        source = str(metadata.get("source") or metadata.get("file_name") or "")
        page_start = metadata.get("page_start")
        chunk_index = metadata.get("chunk_index")
        return (
            document_order.get(source, len(document_order)),
            page_start if isinstance(page_start, int) else 2**31,
            chunk_index if isinstance(chunk_index, int) else 2**31,
            original_order.get(chunk.chunk_id, 2**31),
        )

    def expand(
        self,
        query: str,
        seeds: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        if not seeds:
            return []

        seeds_by_id: Dict[str, RetrievedChunk] = {}
        neighbors_by_id: Dict[str, RetrievedChunk] = {}
        document_order: Dict[str, int] = {}
        for seed_order, seed in enumerate(seeds):
            source = str(
                seed.metadata.get("source")
                or seed.metadata.get("file_name")
                or seed.doc_id
                or ""
            )
            document_order.setdefault(source, seed_order)
            seeds_by_id.setdefault(seed.chunk_id, seed)
            for neighbor in self._neighbors(seed):
                if neighbor.chunk_id not in seeds_by_id:
                    neighbors_by_id.setdefault(neighbor.chunk_id, neighbor)
        for seed_id in seeds_by_id:
            neighbors_by_id.pop(seed_id, None)

        original_order = {
            chunk.chunk_id: index for index, chunk in enumerate(seeds)
        }
        selected: List[RetrievedChunk] = []
        for seed in seeds_by_id.values():
            if self._prompt_fits(query, [*selected, seed]):
                selected.append(seed)
                continue
            if seed.metadata.get("chunk_has_table"):
                fitted_table = self._fit_table(query, selected, seed)
                if fitted_table is not None:
                    selected.append(fitted_table)
                    continue
            logger.warning(
                "Omitting ranked context seed %s because it cannot fit the configured prompt budget (%s tokens).",
                seed.chunk_id,
                self.config.max_prompt_tokens,
            )

        ordered_neighbors = sorted(
            neighbors_by_id.values(),
            key=lambda chunk: self._document_order_key(
                chunk,
                document_order,
                original_order,
            ),
        )
        for chunk in ordered_neighbors:
            if self._prompt_fits(query, [*selected, chunk]):
                selected.append(chunk)
                continue
            if chunk.metadata.get("chunk_has_table"):
                fitted_table = self._fit_table(query, selected, chunk)
                if fitted_table is not None:
                    selected.append(fitted_table)

        return sorted(
            selected,
            key=lambda chunk: self._document_order_key(
                chunk,
                document_order,
                original_order,
            ),
        )


__all__ = ["ContextExpander", "create_context_tokenizer"]
