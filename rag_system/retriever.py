"""Retrieval pipeline: embed query → search vector store."""

from dataclasses import dataclass

from .embeddings import Embedder
from .errors import RetrievalError
from .vector_store import SearchResult, VectorStore


@dataclass(frozen=True)
class RetrievalResult:
    """Result of a retrieval operation."""

    query: str
    chunks: list[SearchResult]


class Retriever:
    """Handles the retrieval workflow: embed query → search → return chunks."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self._store = vector_store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Embed query, search vector store, return top-k chunks.

        Raises RetrievalError if embedding or search fails.
        """
        try:
            query_vec = self._embedder.embed_query(query)
        except Exception as exc:
            raise RetrievalError(f"query embedding failed: {exc}") from exc

        try:
            chunks = self._store.search(query_vec, k=top_k)
        except Exception as exc:
            raise RetrievalError(f"vector search failed: {exc}") from exc

        return RetrievalResult(query=query, chunks=chunks)
