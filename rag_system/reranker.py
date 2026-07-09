"""Cross-encoder reranker for improving retrieval precision.

A cross-encoder (e.g. BAAI/bge-reranker-base) scores (query, passage) pairs
jointly, which is far more accurate than bi-encoder cosine similarity for
ranking — at the cost of being slower. The retriever therefore pulls a
larger candidate set with the bi-encoder and lets the cross-encoder re-score
and trim to the final top_k.
"""

import logging
from typing import Any

from .errors import RetrievalError
from .vector_store import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker backed by sentence-transformers CrossEncoder.

    The model is lazy-loaded on first rerank call so importing this module
    does not pull in PyTorch until reranking is actually requested.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise RetrievalError(
                "sentence-transformers is not installed (required for reranking). "
                "Install it: pip install sentence-transformers"
            )
        try:
            self._model = CrossEncoder(self._model_name, device=self._device)
            logger.info("loaded reranker model: %s", self._model_name)
        except Exception as exc:
            raise RetrievalError(
                f"failed to load reranker model '{self._model_name}': {exc}"
            ) from exc

    def rerank(
        self,
        query: str,
        chunks: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Re-score chunks against the query and return the top_k.

        Returns the input unchanged if empty. Each chunk's ``score`` is
        replaced with the cross-encoder score so callers see the ranking
        confidence.
        """
        if not chunks:
            return []
        self._ensure_loaded()
        pairs = [(query, c.text) for c in chunks]
        try:
            scores = self._model.predict(pairs)  # type: ignore[union-attr]
        except Exception as exc:
            raise RetrievalError(f"reranker predict failed: {exc}") from exc

        # Defensive ravel: CrossEncoder.predict returns a 1-D array today, but
        # flatten in case a future version returns (n, 1).
        try:
            import numpy as _np

            scores = _np.asarray(scores).ravel()
        except ImportError:
            pass

        scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [
            SearchResult(
                text=c.text,
                score=float(s),
                source=c.source,
                metadata=c.metadata,
            )
            for c, s in scored[:top_k]
        ]
