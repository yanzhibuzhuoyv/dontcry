"""rag_system: local RAG pipeline with FAISS + dual embedding support."""

__version__ = "0.1.0"

from .config import (
    ChunkingConfig,
    EmbeddingConfig,
    LLMGenerationConfig,
    RAGConfig,
    load_rag_config,
)
from .errors import ConfigurationError, RAGSystemError
from .rag import RAGSystem

__all__ = [
    "RAGSystem",
    "RAGConfig",
    "EmbeddingConfig",
    "LLMGenerationConfig",
    "ChunkingConfig",
    "load_rag_config",
    "RAGSystemError",
    "ConfigurationError",
]
