"""rag_system: local RAG pipeline with FAISS + dual embedding support."""

__version__ = "0.1.0"

from .config import (
    ChunkingConfig,
    EmbeddingConfig,
    LLMGenerationConfig,
    RAGConfig,
    RetrievalConfig,
    load_rag_config,
)
from .errors import ConfigurationError, RAGSystemError

# RAGSystem pulls in heavy deps (numpy, faiss, openai, sentence-transformers)
# transitively. It is lazy-loaded via __getattr__ so that *directly* importing
# a lightweight submodule (e.g. ``from rag_system.bm25 import BM25Index`` or
# ``from rag_system.splitter import TextSplitter``) does not require those deps
# — important for unit tests and fast CLI startup paths that only need part of
# the package. Note: ``from rag_system import RAGSystem`` still triggers the
# heavy import (that is expected, since RAGSystem itself needs them).


def __getattr__(name: str):
    if name == "RAGSystem":
        from .rag import RAGSystem

        return RAGSystem
    raise AttributeError(f"module 'rag_system' has no attribute {name!r}")


__all__ = [
    "RAGSystem",
    "RAGConfig",
    "RetrievalConfig",
    "EmbeddingConfig",
    "LLMGenerationConfig",
    "ChunkingConfig",
    "load_rag_config",
    "RAGSystemError",
    "ConfigurationError",
]
