"""Retrieval pipeline: embed query → search → optional hybrid fuse / rerank.

Pipeline (each stage opt-in via RetrievalConfig, all off by default so
existing behaviour is unchanged):

    query
      → embed → vector store.search(candidate_count)   # bi-encoder recall
      → [if hybrid] fuse BM25 + cosine over candidates  # keyword boost
      → [if rerank]  cross-encoder re-score → trim      # precision
      → top_k chunks

The candidate count is ``top_k * candidate_multiplier`` when rerank or
hybrid is enabled, so the later (more accurate) stages have room to
re-order. BM25 is built over the candidate set only — cheap and needs no
global index — which improves exact-match ranking within recall but does
not add documents the vector search missed entirely.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .bm25 import BM25Index
from .config import RetrievalConfig
from .embeddings import Embedder
from .errors import RetrievalError
from .reranker import Reranker
from .vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """Result of a retrieval operation."""

    query: str
    chunks: list[SearchResult]


class Retriever:
    """Handles the retrieval workflow with optional hybrid search and rerank."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        retrieval: Optional[RetrievalConfig] = None,
        reranker: Optional[Reranker] = None,
    ):
        self._store = vector_store
        self._embedder = embedder
        self._retrieval = retrieval or RetrievalConfig()
        self._reranker = reranker

    @property
    def retrieval_config(self) -> RetrievalConfig:
        return self._retrieval

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Embed query, search, optionally fuse/rerank, return top-k chunks.

        Raises RetrievalError if embedding or search fails.
        """
        cfg = self._retrieval
        use_rerank = cfg.reranker_enabled and self._reranker is not None
        use_hybrid = cfg.hybrid_enabled
        # Pull a wider candidate set when a later re-scoring stage is active.
        candidates = (
            top_k * cfg.candidate_multiplier
            if (use_rerank or use_hybrid)
            else top_k
        )

        try:
            query_vec = self._embedder.embed_query(query)
        except Exception as exc:
            raise RetrievalError(f"query embedding failed: {exc}") from exc

        try:
            chunks = self._store.search(query_vec, k=candidates)
        except Exception as exc:
            raise RetrievalError(f"vector search failed: {exc}") from exc

        if use_hybrid and chunks:
            chunks = self._hybrid_fuse(query, chunks)

        if use_rerank:
            if not chunks:
                return RetrievalResult(query=query, chunks=[])
            try:
                chunks = self._reranker.rerank(query, chunks, top_k)  # type: ignore[union-attr]
            except RetrievalError as exc:
                # Reranker failure is non-fatal — fall back to vector order.
                logger.warning("reranker failed, using vector order: %s", exc)
                chunks = chunks[:top_k]
        else:
            chunks = chunks[:top_k]

        return RetrievalResult(query=query, chunks=chunks)

    def _hybrid_fuse(
        self, query: str, chunks: list[SearchResult]
    ) -> list[SearchResult]:
        """Re-score candidate chunks by fusing BM25 and cosine similarity.

        ``hybrid_alpha`` weights BM25; ``1 - alpha`` weights the vector
        score. BM25 scores are min-max normalised to [0, 1] so they are
        comparable with cosine (already in [-1, 1] on normalised vectors,
        effectively [0, 1] for relevant docs).

        If ``adaptive_alpha`` is enabled, alpha is chosen per-query by
        length as a heuristic: short queries (keywords/titles) keep more
        vector weight, long descriptive queries lean on BM25 keyword
        matching. NOTE: this length-based heuristic is NOT validated by the
        alpha-tuning benchmark (which only swept a static alpha); the
        threshold and two alphas are empirical guesses. Leave adaptive_alpha
        off (the default) unless you have measured it helps on your own data.

        If BM25 produces no signal at all (every candidate scores 0 — common
        for short queries with no keyword overlap), the original vector
        ordering and scores are returned unchanged. Fusing in that case would
        only scale every score by ``(1 - alpha)`` without reordering, which
        distorts absolute scores (and any downstream threshold/rerank input)
        for no benefit.
        """
        bm25 = BM25Index([c.text for c in chunks])
        raw_bm25 = bm25.scores(query)
        max_b = max(raw_bm25) if raw_bm25 else 0.0
        if max_b <= 0.0:
            # No BM25 signal — keep vector scores untouched.
            return chunks

        alpha = self._alpha_for_query(query)
        fused: list[SearchResult] = []
        for chunk, bs in zip(chunks, raw_bm25):
            score = alpha * (bs / max_b) + (1 - alpha) * chunk.score
            fused.append(
                SearchResult(
                    text=chunk.text,
                    score=float(score),
                    source=chunk.source,
                    metadata=chunk.metadata,
                )
            )
        fused.sort(key=lambda c: c.score, reverse=True)
        return fused

    def _alpha_for_query(self, query: str) -> float:
        """Pick the BM25/vector fusion weight for this query.

        With adaptive_alpha off, returns the static ``hybrid_alpha``. With it
        on, short queries (<= short_query_threshold chars) use
        ``short_query_alpha`` and longer ones use ``long_query_alpha``.
        """
        cfg = self._retrieval
        if not cfg.adaptive_alpha:
            return cfg.hybrid_alpha
        if len(query) <= cfg.short_query_threshold:
            return cfg.short_query_alpha
        return cfg.long_query_alpha
