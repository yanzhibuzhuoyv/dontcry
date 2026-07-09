"""Unit tests for VectorStore add/search/remove.

Requires faiss-cpu + numpy (skipped automatically if unavailable). Uses a
deterministic fake embedder so tests run offline without any model download.
"""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("faiss")

from rag_system.documents import Chunk  # noqa: E402
from rag_system.vector_store import VectorStore  # noqa: E402

_DIM = 4


def _vec(seed: int) -> list[float]:
    """Deterministic unit-ish vector from an int seed."""
    import math

    vals = [math.sin(seed + i) for i in range(_DIM)]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def _chunk(text: str, source: str, idx: int, h: str = "") -> Chunk:
    return Chunk(text=text, source=source, chunk_index=idx, metadata={"content_hash": h})


def test_add_and_count():
    store = VectorStore(_DIM)
    assert store.count == 0
    chunks = [_chunk("a", "f1", 0, "h1"), _chunk("b", "f1", 1, "h1")]
    store.add_documents(chunks, [_vec(1), _vec(2)])
    assert store.count == 2
    assert store.has_source("f1")
    assert "f1" in store.sources


def test_file_hash_o1():
    store = VectorStore(_DIM)
    store.add_documents([_chunk("a", "f1", 0, "hash123")], [_vec(1)])
    assert store.file_hash("f1") == "hash123"
    assert store.file_hash("missing") is None


def test_search_returns_relevant():
    store = VectorStore(_DIM)
    store.add_documents(
        [_chunk("alpha", "f1", 0), _chunk("beta", "f2", 0)],
        [_vec(1), _vec(2)],
    )
    results = store.search(_vec(1), k=2)
    assert len(results) == 2
    # The vector matching _vec(1) (seed 1) should be most similar to itself.
    assert results[0].source == "f1"


def test_remove_source():
    store = VectorStore(_DIM)
    store.add_documents(
        [_chunk("a", "f1", 0, "h1"), _chunk("b", "f2", 0, "h2")],
        [_vec(1), _vec(2)],
    )
    removed = store.remove_source("f1")
    assert removed == 1
    assert not store.has_source("f1")
    assert store.file_hash("f1") is None
    assert store.count == 1
    # Remaining source still searchable.
    results = store.search(_vec(2), k=5)
    assert len(results) == 1
    assert results[0].source == "f2"


def test_remove_missing_source_returns_zero():
    store = VectorStore(_DIM)
    assert store.remove_source("nope") == 0


def test_remove_source_multi_chunk_and_readd():
    """Removing a multi-chunk source and re-adding keeps id continuity."""
    store = VectorStore(_DIM)
    # f1 has 3 chunks, f2 has 1 chunk.
    store.add_documents(
        [_chunk("a", "f1", 0, "h1"), _chunk("b", "f1", 1, "h1"), _chunk("c", "f1", 2, "h1")],
        [_vec(1), _vec(2), _vec(3)],
    )
    store.add_documents([_chunk("d", "f2", 0, "h2")], [_vec(4)])
    assert store.count == 4

    removed = store.remove_source("f1")
    assert removed == 3
    assert store.count == 1
    assert not store.has_source("f1")

    # Re-add f1 — must not collide with f2's existing vector.
    store.add_documents(
        [_chunk("new", "f1", 0, "h1b")], [_vec(5)]
    )
    assert store.count == 2
    assert store.has_source("f1")
    # Both sources searchable.
    results = store.search(_vec(4), k=5)
    assert any(r.source == "f2" for r in results)


def test_remove_source_then_save_load(tmp_path):
    store = VectorStore(_DIM)
    store.add_documents(
        [_chunk("a", "f1", 0, "h1"), _chunk("b", "f2", 0, "h2")],
        [_vec(1), _vec(2)],
    )
    store.remove_source("f1")
    d = tmp_path / "store"
    store.save(d)
    loaded = VectorStore.load(d)
    assert loaded.count == 1
    assert loaded.has_source("f2")
    assert not loaded.has_source("f1")


def test_save_load_roundtrip(tmp_path):
    store = VectorStore(_DIM)
    store.add_documents(
        [_chunk("alpha", "f1", 0, "h1")], [_vec(1)]
    )
    d = tmp_path / "store"
    store.save(d)
    loaded = VectorStore.load(d)
    assert loaded.count == 1
    assert loaded.dimension == _DIM
    assert loaded.has_source("f1")
    assert loaded.file_hash("f1") == "h1"
    # source_hashes survives the round trip
    assert "f1" in loaded.sources
