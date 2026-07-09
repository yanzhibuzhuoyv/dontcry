"""Unit tests for the Retriever pipeline (hybrid fuse + rerank + fallback).

Uses MagicMock for the vector store and embedder so tests run offline without
faiss/numpy/model downloads. Focuses on pipeline logic: candidate count,
hybrid fusion, rerank fallback, empty-index handling.
"""

from unittest.mock import MagicMock

from rag_system.config import RetrievalConfig
from rag_system.errors import RetrievalError
from rag_system.retriever import Retriever
from rag_system.vector_store import SearchResult


def _chunks(n: int, score_start: float = 0.9) -> list[SearchResult]:
    """n fake chunks with descending vector scores."""
    return [
        SearchResult(
            text=f"doc {i}", score=score_start - i * 0.1,
            source=f"f{i}", metadata={},
        )
        for i in range(n)
    ]


def _mock_store_embedder(chunks_returned: list[SearchResult]):
    store = MagicMock()
    store.search.return_value = chunks_returned
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    return store, embedder


def test_disabled_uses_top_k_candidates():
    chunks = _chunks(5)
    store, embedder = _mock_store_embedder(chunks)
    r = Retriever(store, embedder)  # default config: hybrid/rerank off
    result = r.retrieve("q", top_k=5)
    store.search.assert_called_once_with([0.1, 0.2, 0.3, 0.4], k=5)
    assert len(result.chunks) == 5


def test_empty_store_returns_empty():
    store, embedder = _mock_store_embedder([])
    r = Retriever(store, embedder)
    result = r.retrieve("q", top_k=5)
    assert result.chunks == []


def test_rerank_uses_multiplier_candidates():
    chunks = _chunks(20)
    store, embedder = _mock_store_embedder(chunks)
    reranker = MagicMock()
    reranker.rerank.return_value = _chunks(2, score_start=0.99)
    cfg = RetrievalConfig(reranker_enabled=True, candidate_multiplier=4)
    r = Retriever(store, embedder, retrieval=cfg, reranker=reranker)
    r.retrieve("q", top_k=5)
    # candidates = top_k * multiplier = 5 * 4 = 20
    store.search.assert_called_once_with([0.1, 0.2, 0.3, 0.4], k=20)
    reranker.rerank.assert_called_once()
    # rerank is called positionally as rerank(query, chunks, top_k).
    args = reranker.rerank.call_args
    assert args[0][2] == 5


def test_rerank_fallback_on_failure():
    chunks = _chunks(8)
    store, embedder = _mock_store_embedder(chunks)
    reranker = MagicMock()
    reranker.rerank.side_effect = RetrievalError("model down")
    cfg = RetrievalConfig(reranker_enabled=True, candidate_multiplier=4)
    r = Retriever(store, embedder, retrieval=cfg, reranker=reranker)
    result = r.retrieve("q", top_k=2)
    # Fallback to vector order: top-2 by original score.
    assert len(result.chunks) == 2
    assert result.chunks[0].source == "f0"
    assert result.chunks[1].source == "f1"


def test_hybrid_bm25_zero_keeps_vector_scores():
    """P1 fix: when BM25 finds no signal, vector scores must NOT be scaled."""
    # English chunks + Chinese query → no bigram overlap → BM25 all zero.
    chunks = [
        SearchResult(text="alpha beta gamma", score=0.9, source="f1", metadata={}),
        SearchResult(text="delta epsilon zeta", score=0.8, source="f2", metadata={}),
    ]
    store, embedder = _mock_store_embedder(chunks)
    cfg = RetrievalConfig(hybrid_enabled=True, hybrid_alpha=0.3)
    r = Retriever(store, embedder, retrieval=cfg)
    result = r.retrieve("检索系统", top_k=2)
    # Scores unchanged (not multiplied by 1-alpha=0.7).
    assert result.chunks[0].score == 0.9
    assert result.chunks[1].score == 0.8


def test_hybrid_with_signal_reorders():
    """When BM25 has signal, a matching chunk can be boosted above higher-vector ones."""
    # chunk f1 has the query keyword, f2 does not but higher vector score.
    chunks = [
        SearchResult(text="检索系统", score=0.5, source="f1", metadata={}),
        SearchResult(text="无关内容", score=0.9, source="f2", metadata={}),
    ]
    store, embedder = _mock_store_embedder(chunks)
    cfg = RetrievalConfig(hybrid_enabled=True, hybrid_alpha=0.6)  # favour BM25
    r = Retriever(store, embedder, retrieval=cfg)
    result = r.retrieve("检索", top_k=2)
    # f1 (matches 检索) should now rank first despite lower vector score.
    assert result.chunks[0].source == "f1"


def test_adaptive_alpha_picks_by_query_length():
    """adaptive_alpha uses short_query_alpha for short queries, long for long."""
    store, embedder = _mock_store_embedder(_chunks(5))
    cfg = RetrievalConfig(
        hybrid_enabled=True,
        adaptive_alpha=True,
        short_query_threshold=12,
        short_query_alpha=0.4,
        long_query_alpha=0.6,
    )
    r = Retriever(store, embedder, retrieval=cfg)
    assert r._alpha_for_query("短查询") == 0.4  # 3 chars <= 12
    assert r._alpha_for_query("这是一个比较长的查询超过十二个字") == 0.6  # > 12


def test_adaptive_alpha_off_uses_static_hybrid_alpha():
    """With adaptive_alpha off, _alpha_for_query returns the static hybrid_alpha."""
    store, embedder = _mock_store_embedder(_chunks(5))
    cfg = RetrievalConfig(hybrid_enabled=True, hybrid_alpha=0.5, adaptive_alpha=False)
    r = Retriever(store, embedder, retrieval=cfg)
    assert r._alpha_for_query("anything") == 0.5


def test_adaptive_alpha_changes_order_for_short_vs_long_query():
    """End-to-end: adaptive_alpha must actually affect _hybrid_fuse ordering.

    f1 matches the query keyword but has a low vector score; f2 has a high
    vector score but no keyword match. With short_query_alpha low (vector
    favoured) f2 wins; with long_query_alpha high (BM25 favoured) f1 wins.
    This guards against the feature silently dying if _hybrid_fuse stops
    calling _alpha_for_query.
    """
    chunks = [
        SearchResult(text="检索系统", score=0.1, source="f1", metadata={}),
        SearchResult(text="无关内容", score=0.9, source="f2", metadata={}),
    ]
    store, embedder = _mock_store_embedder(chunks)
    cfg = RetrievalConfig(
        hybrid_enabled=True,
        adaptive_alpha=True,
        short_query_threshold=12,
        short_query_alpha=0.3,  # vector-favoured
        long_query_alpha=0.7,   # BM25-favoured
    )
    r = Retriever(store, embedder, retrieval=cfg)

    # Short query (4 chars): vector weight high → f2 (0.9) ranks first.
    short_result = r.retrieve("检索系统", top_k=2)
    assert short_result.chunks[0].source == "f2"

    # Long query (>12 chars): BM25 weight high → f1 (keyword match) ranks first.
    long_result = r.retrieve("检索系统的实现方案与挑战探讨", top_k=2)
    assert long_result.chunks[0].source == "f1"
