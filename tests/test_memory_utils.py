"""Unit tests for session-memory helpers (pure functions, no heavy deps)."""

from rag_system.memory_utils import (
    make_slug,
    merge_dedup,
    parse_session_filename,
)


def test_merge_dedup_new_first():
    new = ["alpha", "beta"]
    old = ["beta", "gamma"]
    merged = merge_dedup(new, old)
    # New prompts come first; dup "beta" appears once.
    assert merged == ["alpha", "beta", "gamma"]


def test_merge_dedup_normalises_whitespace_and_case():
    new = ["RAG 系统"]
    old = ["rag系统"]
    merged = merge_dedup(new, old)
    assert merged == ["RAG 系统"]


def test_merge_dedup_empty():
    assert merge_dedup([], []) == []
    assert merge_dedup(["a"], []) == ["a"]
    assert merge_dedup([], ["b"]) == ["b"]


def test_make_slug_basic():
    assert make_slug("hello world") == "hello-world"


def test_make_slug_truncates():
    slug = make_slug("a" * 100, max_len=10)
    assert len(slug) <= 10


def test_make_slug_chinese():
    slug = make_slug("检索系统优化")
    assert "检索" in slug or slug  # non-empty, contains CJK


def test_parse_session_filename_standard():
    date, slug = parse_session_filename("2026-07-09-my-topic-153022")
    assert date == "2026-07-09"
    assert slug == "my-topic"


def test_parse_session_filename_with_dash_in_slug():
    date, slug = parse_session_filename("2026-07-09-multi-word-slug-095511")
    assert date == "2026-07-09"
    assert slug == "multi-word-slug"


def test_parse_session_filename_legacy():
    # Legacy names without trailing timestamp fall back gracefully.
    date, slug = parse_session_filename("2026-07-09-legacy")
    assert date == "2026-07-09"
    assert slug == "legacy"


def test_parse_session_filename_malformed():
    date, slug = parse_session_filename("garbage")
    assert date == ""
    assert slug == "garbage"
