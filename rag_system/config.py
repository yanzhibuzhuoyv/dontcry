"""Environment-variable driven configuration for the RAG system.

All config objects are frozen dataclasses — create a new one to change settings.

Loads .env from the project root on import (python-dotenv required, but optional).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .errors import ConfigurationError

# Auto-load .env from project root (two levels up from this file)
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv not installed — env vars must be set manually


_load_dotenv()


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
class RAGConfig:
    """Top-level RAG configuration aggregating all sub-configs."""

    embedding: EmbeddingConfig
    llm: LLMGenerationConfig
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
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
}


def _env(key: str) -> str:
    """Read an environment variable, falling back to defaults table."""
    return os.environ.get(key, _DEFAULTS.get(key, ""))


def load_rag_config() -> RAGConfig:
    """Build a RAGConfig from RAG_* environment variables.

    Raises ConfigurationError if required values are missing.
    """
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
        temperature=float(_env("RAG_LLM_TEMPERATURE")),
        max_tokens=int(_env("RAG_LLM_MAX_TOKENS")),
    )

    chunking = ChunkingConfig(
        chunk_size=int(_env("RAG_CHUNK_SIZE")),
        chunk_overlap=int(_env("RAG_CHUNK_OVERLAP")),
    )

    return RAGConfig(
        embedding=embedding,
        llm=llm,
        chunking=chunking,
        top_k=int(_env("RAG_TOP_K")),
        vector_store_dir=_env("RAG_VECTOR_STORE_DIR"),
    )
