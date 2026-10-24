import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Nạp các biến môi trường từ file .env
load_dotenv()

# Đường dẫn thư mục gốc dự án (vietnamese-rag)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------------------
# 1. Sub-configs
# ------------------------------------------------------------------------------

class AppMetaConfig(BaseModel):
    name: str = "vietnamese-rag"
    version: str = "0.1.0"
    env: str = "development"
    framework: str = "llamaindex"
    seed: int = 42
    log_level: str = "INFO"


class PDFOCRConfig(BaseModel):
    enabled: bool = True
    languages: str = "vie+eng"
    dpi: int = 300
    min_text_chars: int = 40
    tesseract_cmd: Optional[str] = None


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    supported_extensions: List[str] = [".pdf", ".docx", ".txt"]
    pdf_ocr: PDFOCRConfig = Field(default_factory=PDFOCRConfig)

    @property
    def raw_path(self) -> Path:
        return PROJECT_ROOT / self.raw_dir

    @property
    def processed_path(self) -> Path:
        return PROJECT_ROOT / self.processed_dir


class RecursiveChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: List[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    )


class StructuredChunkingConfig(BaseModel):
    max_tokens: int = 512
    overlap_tokens: int = 64


class SemanticChunkingConfig(BaseModel):
    buffer_size: int = 1
    breakpoint_percentile_threshold: int = 95


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    recursive: RecursiveChunkingConfig = Field(default_factory=RecursiveChunkingConfig)
    structured: StructuredChunkingConfig = Field(default_factory=StructuredChunkingConfig)
    semantic: SemanticChunkingConfig = Field(default_factory=SemanticChunkingConfig)


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    provider: str = "huggingface"
    dimension: int = 1024
    device: str = "cuda"
    batch_size: int = 16
    normalize_embeddings: bool = True
    max_length: int = 8192


class VectorStoreConfig(BaseModel):
    type: str = "qdrant"
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    collection_name: str = "vietnamese_docs"
    dense_vector_name: Optional[str] = None
    distance: str = "Cosine"
    timeout: int = 30
    api_key: Optional[str] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class HybridSearchConfig(BaseModel):
    enabled: bool = True
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    candidate_k: Optional[int] = None
    candidate_multiplier: int = 5
    min_candidates: int = 10
    rrf_k: int = 60


class ContextExpansionConfig(BaseModel):
    enabled: bool = True
    neighbor_window: int = Field(default=1, ge=0)
    max_prompt_tokens: int = Field(default=8192, ge=1)
    tokenizer_name: Optional[str] = None


class BGERerankerConfig(BaseModel):
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    batch_size: int = 8
    max_length: int = 1024


class CohereRerankerConfig(BaseModel):
    model_name: str = "rerank-multilingual-v3.0"
    api_key: Optional[str] = None


class RerankerConfig(BaseModel):
    enabled: bool = True
    provider: str = "bge"
    top_n: int = 3
    bge: BGERerankerConfig = Field(default_factory=BGERerankerConfig)
    cohere: CohereRerankerConfig = Field(default_factory=CohereRerankerConfig)


class RetrievalConfig(BaseModel):
    top_k: int = 10
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    context_expansion: ContextExpansionConfig = Field(default_factory=ContextExpansionConfig)


class GroqLLMConfig(BaseModel):
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 1500
    timeout: int = 60
    api_key: Optional[str] = None


class OllamaLLMConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:14b"
    temperature: float = 0.1
    max_tokens: int = 1500
    timeout: int = 120


class LLMConfig(BaseModel):
    default_provider: str = "groq"
    groq: GroqLLMConfig = Field(default_factory=GroqLLMConfig)
    ollama: OllamaLLMConfig = Field(default_factory=OllamaLLMConfig)


class PromptsConfig(BaseModel):
    qa_template_path: str = "configs/prompts/qa_prompt.yaml"
    eval_template_path: str = "configs/prompts/eval_prompt.yaml"

    @property
    def qa_path(self) -> Path:
        return PROJECT_ROOT / self.qa_template_path

    @property
    def eval_path(self) -> Path:
        return PROJECT_ROOT / self.eval_template_path


class JudgeLLMConfig(BaseModel):
    provider: str = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    api_key: Optional[str] = None


class JudgeEmbeddingConfig(BaseModel):
    provider: str = "huggingface"
    model_name: str = "BAAI/bge-m3"
    device: str = "cuda"


class EvaluationConfig(BaseModel):
    testset_path: str = "data/eval/testset.jsonl"
    metrics: List[str] = Field(
        default_factory=lambda: [
            "context_recall",
            "context_precision",
            "faithfulness",
            "answer_relevancy",
        ]
    )
    judge_llm: JudgeLLMConfig = Field(default_factory=JudgeLLMConfig)
    judge_embedding: JudgeEmbeddingConfig = Field(default_factory=JudgeEmbeddingConfig)

    @property
    def testset_file_path(self) -> Path:
        return PROJECT_ROOT / self.testset_path


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True


# ------------------------------------------------------------------------------
# 2. Main Root Configuration
# ------------------------------------------------------------------------------

class AppConfig(BaseModel):
    app: AppMetaConfig = Field(default_factory=AppMetaConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)


# ------------------------------------------------------------------------------
# 3. Loader Functions
# ------------------------------------------------------------------------------

def load_config(config_path: Optional[Path | str] = None) -> AppConfig:
    """Nạp cấu hình từ file YAML và kết hợp với biến môi trường (.env)."""
    if config_path is None:
        target_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        target_path = Path(config_path)

    raw_dict = {}
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

    config = AppConfig.model_validate(raw_dict)

    # Nạp Secret Keys & overrides từ biến môi trường
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        config.llm.groq.api_key = groq_key

    # API key riêng cho LLM Đánh giá (Judge LLM)
    eval_key = os.getenv("EVAL_GROQ_API_KEY") or os.getenv("JUDGE_API_KEY")
    if eval_key:
        config.evaluation.judge_llm.api_key = eval_key
    elif groq_key:
        config.evaluation.judge_llm.api_key = groq_key

    cohere_key = os.getenv("COHERE_API_KEY")
    if cohere_key:
        config.retrieval.reranker.cohere.api_key = cohere_key

    qdrant_key = os.getenv("QDRANT_API_KEY")
    if qdrant_key:
        config.vector_store.api_key = qdrant_key

    # Cho phép override host/port Qdrant qua .env nếu có
    env_qdrant_host = os.getenv("QDRANT_HOST")
    if env_qdrant_host:
        config.vector_store.host = env_qdrant_host

    env_qdrant_port = os.getenv("QDRANT_PORT")
    if env_qdrant_port:
        config.vector_store.port = int(env_qdrant_port)

    env_ollama_url = os.getenv("OLLAMA_BASE_URL")
    if env_ollama_url:
        config.llm.ollama.base_url = env_ollama_url

    return config


@lru_cache()
def _get_cached_config() -> AppConfig:
    return load_config()


def get_config(reload: bool = False) -> AppConfig:
    """Trả về AppConfig. Nếu reload=True, xóa cache và nạp lại cấu hình mới nhất từ file YAML."""
    if reload:
        _get_cached_config.cache_clear()
    return _get_cached_config()


if __name__ == "__main__":
    cfg = get_config()
    print("=" * 60)
    print(f"App Name: {cfg.app.name} (v{cfg.app.version})")
    print(f"Framework: {cfg.app.framework}")
    print(f"Vector DB: {cfg.vector_store.url} -> Collection: {cfg.vector_store.collection_name}")
    print(f"Embedding: {cfg.embedding.model_name} (dim: {cfg.embedding.dimension})")
    print(f"LLM Provider: {cfg.llm.default_provider} -> Model: {cfg.llm.groq.model_name}")
    print(f"Reranker: {cfg.retrieval.reranker.provider} -> Model: {cfg.retrieval.reranker.bge.model_name}")
    print("=" * 60)
    print("Config validation passed successfully!")
