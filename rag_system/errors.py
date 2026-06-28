"""Custom exception hierarchy for the RAG system."""


class RAGSystemError(Exception):
    """Base exception for all RAG system errors."""


class ConfigurationError(RAGSystemError):
    """Missing or invalid configuration (env vars, file paths, etc.)."""


class IngestionError(RAGSystemError):
    """Document loading or conversion failed."""


class EmbeddingError(RAGSystemError):
    """Embedding generation failed (local model or API)."""


class VectorStoreError(RAGSystemError):
    """FAISS read/write/search operation failed."""


class RetrievalError(RAGSystemError):
    """Retrieval pipeline failed (embedding + search)."""


class GenerationError(RAGSystemError):
    """LLM generation failed."""
