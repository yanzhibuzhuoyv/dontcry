"""Unit tests for the BM25 hybrid-search index.

Pure Python, no external deps. Verifies tokenisation (CJK bigrams + ASCII
words) and that exact keyword matches rank above non-matches.
"""

from rag_system.bm25 import BM25Index, _tokenize


def test_tokenize_ascii_words():
    toks = _tokenize("hello world foo")
    assert toks == ["hello", "world", "foo"]


def test_tokenize_cjk_bigrams():
    toks = _tokenize("检索系统")
    # CJK bigrams of 检索系统: 检索, 索系, 系统
    assert "检索" in toks
    assert "系统" in toks
    assert "索系" in toks


def test_tokenize_mixed():
    toks = _tokenize("RAG 检索")
    assert "rag" in toks
    assert "检索" in toks


def test_bm25_exact_match_beats_nonmatch():
    corpus = [
        "本文讨论向量检索的原理",
        "今天天气不错适合散步",
        "检索系统优化方案",
    ]
    idx = BM25Index(corpus)
    scores = idx.scores("检索")
    # Docs 0 and 2 contain 检索, doc 1 does not.
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]
    # Matching docs must have a positive score (guards against all-zero bugs).
    assert scores[0] > 0
    assert scores[2] > 0


def test_bm25_empty_query():
    idx = BM25Index(["some text", "more text"])
    assert idx.scores("") == [0.0, 0.0]


def test_bm25_empty_corpus():
    idx = BM25Index([])
    assert idx.scores("anything") == []


def test_bm25_no_match_returns_zeros():
    idx = BM25Index(["苹果", "香蕉"])
    scores = idx.scores("zzz")
    assert scores == [0.0, 0.0]
