"""Environment-variable driven configuration for the RAG system.

All config objects are frozen dataclasses — create a new one to change settings.

Loads .env from the project root on import (python-dotenv required, but optional).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .errors import ConfigurationError

# Auto-load .env from project root (two levels up from this file).
# Done lazily inside load_rag_config() rather than at import time so that
# merely importing this module has no side effects on os.environ (which
# previously made test isolation harder).
_dotenv_loaded = False


def _load_dotenv() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv not installed — env vars must be set manually


# ---------------------------------------------------------------------------
# Individual config objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingConfig:
    """Controls which embedder to use and how to configure it."""

    provider: Literal["local", "api"]
    model: str
    base_url: str = ""
    api_key: str = ""
    device: str = "cpu"

    def __post_init__(self):
        if self.provider == "api" and not self.base_url:
            raise ConfigurationError("embedding base_url is required when provider='api'")
        if self.provider == "api" and not self.api_key:
            raise ConfigurationError("embedding api_key is required when provider='api'")


@dataclass(frozen=True)
class LLMGenerationConfig:
    """OpenAI-compatible chat completion parameters."""

    model: str
    base_url: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2048

    def __post_init__(self):
        if not self.api_key:
            raise ConfigurationError("LLM api_key is required")
        if not self.base_url:
            raise ConfigurationError("LLM base_url is required")
        if not (0 <= self.temperature <= 2):
            raise ConfigurationError("temperature must be in [0, 2]")


@dataclass(frozen=True)
class ChunkingConfig:
    """Text splitting parameters."""

    chunk_size: int = 512
    chunk_overlap: int = 128

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ConfigurationError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("chunk_overlap must be < chunk_size")


@dataclass(frozen=True)
class RetrievalConfig:
    """Controls the retrieval pipeline: hybrid search and reranking.

    Both are opt-in (disabled by default) so existing behaviour is unchanged
    unless the user explicitly enables them via env vars.

    Alpha tuning (500-doc / 1000-query benchmark with bge-small-zh-v1.5):
    the previous default alpha=0.3 under-weighted BM25. Sweeping alpha showed
    fuzzy-query MRR climbs monotonically from 0.23 (alpha=0) to 0.50
    (alpha=1), while exact-query MRR stays ~0.94 across the range. The new
    default 0.5 is a robust middle ground.

    Caveat: in that benchmark alpha=1.0 (pure BM25) was the global optimum —
    an anomaly caused by the synthetic queries being verbatim substrings of
    the source documents, which favours literal bigram matching. Real user
    queries (paraphrases, conceptual questions) will not favour BM25 this
    strongly, so 0.5 rather than 1.0 is the default. The adaptive_alpha
    option is an unvalidated heuristic — measure on your own data before
    enabling it.
    """

    hybrid_enabled: bool = False
    # BM25 weight in the fused score; (1 - alpha) is the vector weight.
    hybrid_alpha: float = 0.5
    # When True, override hybrid_alpha per-query by query length: short
    # queries (likely keywords/titles) keep more vector weight, long queries
    # (likely descriptive snippets) lean on BM25 keyword matching.
    adaptive_alpha: bool = False
    short_query_threshold: int = 12
    short_query_alpha: float = 0.4
    long_query_alpha: float = 0.6
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    # When reranking, retrieve top_k * multiplier candidates first, then let
    # the cross-encoder re-score and trim to top_k.
    candidate_multiplier: int = 4

    def __post_init__(self):
        for name, val in (
            ("hybrid_alpha", self.hybrid_alpha),
            ("short_query_alpha", self.short_query_alpha),
            ("long_query_alpha", self.long_query_alpha),
        ):
            if not (0.0 <= val <= 1.0):
                raise ConfigurationError(f"{name} must be in [0, 1]")
        if self.candidate_multiplier < 1:
            raise ConfigurationError("candidate_multiplier must be >= 1")
        if self.short_query_threshold < 1:
            raise ConfigurationError("short_query_threshold must be >= 1")


@dataclass(frozen=True)
class RAGConfig:
    """Top-level RAG configuration aggregating all sub-configs."""

    embedding: EmbeddingConfig
    llm: LLMGenerationConfig
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    top_k: int = 5
    vector_store_dir: str = "./rag_index"

    def __post_init__(self):
        if self.top_k <= 0:
            raise ConfigurationError("top_k must be positive")


# ---------------------------------------------------------------------------
# Environment variable loading
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "RAG_EMBEDDING_PROVIDER": "local",
    "RAG_EMBEDDING_MODEL": "BAAI/bge-small-zh-v1.5",
    "RAG_EMBEDDING_BASE_URL": "https://api.deepseek.com/v1",
    "RAG_EMBEDDING_API_KEY": "",
    "RAG_EMBEDDING_DEVICE": "cpu",
    "RAG_LLM_MODEL": "deepseek-chat",
    "RAG_LLM_BASE_URL": "https://api.deepseek.com/v1",
    "RAG_LLM_API_KEY": "",
    "RAG_LLM_TEMPERATURE": "0.7",
    "RAG_LLM_MAX_TOKENS": "2048",
    "RAG_CHUNK_SIZE": "512",
    "RAG_CHUNK_OVERLAP": "128",
    "RAG_TOP_K": "5",
    "RAG_VECTOR_STORE_DIR": "./rag_index",
    "RAG_HYBRID_ENABLED": "false",
    "RAG_HYBRID_ALPHA": "0.5",
    "RAG_ADAPTIVE_ALPHA": "false",
    "RAG_SHORT_QUERY_THRESHOLD": "12",
    "RAG_SHORT_QUERY_ALPHA": "0.4",
    "RAG_LONG_QUERY_ALPHA": "0.6",
    "RAG_RERANKER_ENABLED": "false",
    "RAG_RERANKER_MODEL": "BAAI/bge-reranker-base",
    "RAG_CANDIDATE_MULTIPLIER": "4",
}


def _env(key: str) -> str:
    """Read an environment variable, falling back to defaults table."""
    return os.environ.get(key, _DEFAULTS.get(key, ""))


def _env_int(key: str) -> int:
    """Read an integer env var, raising ConfigurationError on bad input."""
    raw = _env(key)
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(f"{key} must be an integer, got '{raw}'")


def _env_float(key: str) -> float:
    """Read a float env var, raising ConfigurationError on bad input."""
    raw = _env(key)
    try:
        return float(raw)
    except ValueError:
        raise ConfigurationError(f"{key} must be a number, got '{raw}'")


def _env_bool(key: str) -> bool:
    """Read a boolean env var. Accepts 1/true/yes/on/y/是 (case-insensitive)."""
    raw = _env(key).strip().lower()
    return raw in ("1", "true", "yes", "on", "y", "是")


def load_rag_config() -> RAGConfig:
    """Build a RAGConfig from RAG_* environment variables.

    Raises ConfigurationError if required values are missing or malformed.
    """
    _load_dotenv()

    embedding_provider = _env("RAG_EMBEDDING_PROVIDER").strip().lower()
    if embedding_provider not in ("local", "api"):
        raise ConfigurationError(
            f"RAG_EMBEDDING_PROVIDER must be 'local' or 'api', got '{embedding_provider}'"
        )

    embedding_api_key = _env("RAG_EMBEDDING_API_KEY")
    if embedding_provider == "api" and not embedding_api_key:
        embedding_api_key = _env("RAG_LLM_API_KEY")

    embedding = EmbeddingConfig(
        provider=embedding_provider,  # type: ignore[arg-type]
        model=_env("RAG_EMBEDDING_MODEL"),
        base_url=_env("RAG_EMBEDDING_BASE_URL") if embedding_provider == "api" else "",
        api_key=embedding_api_key if embedding_provider == "api" else "",
        device=_env("RAG_EMBEDDING_DEVICE"),
    )

    llm = LLMGenerationConfig(
        model=_env("RAG_LLM_MODEL"),
        base_url=_env("RAG_LLM_BASE_URL"),
        api_key=_env("RAG_LLM_API_KEY"),
        temperature=_env_float("RAG_LLM_TEMPERATURE"),
        max_tokens=_env_int("RAG_LLM_MAX_TOKENS"),
    )

    chunking = ChunkingConfig(
        chunk_size=_env_int("RAG_CHUNK_SIZE"),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP"),
    )

    retrieval = RetrievalConfig(
        hybrid_enabled=_env_bool("RAG_HYBRID_ENABLED"),
        hybrid_alpha=_env_float("RAG_HYBRID_ALPHA"),
        adaptive_alpha=_env_bool("RAG_ADAPTIVE_ALPHA"),
        short_query_threshold=_env_int("RAG_SHORT_QUERY_THRESHOLD"),
        short_query_alpha=_env_float("RAG_SHORT_QUERY_ALPHA"),
        long_query_alpha=_env_float("RAG_LONG_QUERY_ALPHA"),
        reranker_enabled=_env_bool("RAG_RERANKER_ENABLED"),
        reranker_model=_env("RAG_RERANKER_MODEL"),
        candidate_multiplier=_env_int("RAG_CANDIDATE_MULTIPLIER"),
    )

    return RAGConfig(
        embedding=embedding,
        llm=llm,
        chunking=chunking,
        retrieval=retrieval,
        top_k=_env_int("RAG_TOP_K"),
        vector_store_dir=_env("RAG_VECTOR_STORE_DIR"),
    )
