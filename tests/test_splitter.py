"""Unit tests for the recursive text splitter (Chinese-punctuation aware).

These run without numpy/faiss — splitter only depends on the standard library
plus the lightweight config/documents modules.
"""

from rag_system.splitter import TextSplitter


def test_split_empty():
    s = TextSplitter(chunk_size=64, chunk_overlap=16)
    assert s.split("") == []
    assert s.split("   ") == []


def test_split_single_short_text():
    s = TextSplitter(chunk_size=512)
    chunks = s.split("这是一段短文本。")
    assert chunks == ["这是一段短文本。"]


def test_chinese_punctuation_split():
    s = TextSplitter(chunk_size=24, chunk_overlap=0)
    text = "句一。句二。句三。句四。"
    chunks = s.split(text)
    # Each sentence "句X。" is 3 chars; chunk_size=24 fits several.
    assert len(chunks) >= 1
    # No chunk should exceed chunk_size.
    for c in chunks:
        assert len(c) <= 24


def test_overlap_produces_shared_content():
    """The P0 fix: adjacent chunks must carry overlap from the previous tail."""
    s = TextSplitter(chunk_size=24, chunk_overlap=12)
    text = "".join(f"句{i}内容。" for i in range(1, 11))
    chunks = s.split(text)
    assert len(chunks) >= 2

    def _shared_prefix_suffix(a: str, b: str, min_n: int = 3) -> int:
        """Length of the longest suffix of *a* that is a prefix of *b*."""
        max_n = min(len(a), len(b), 16)
        for n in range(max_n, min_n - 1, -1):
            if a[-n:] == b[:n]:
                return n
        return 0

    found_overlap = False
    for i in range(len(chunks) - 1):
        if _shared_prefix_suffix(chunks[i], chunks[i + 1]) >= 3:
            found_overlap = True
            break
    assert found_overlap, f"no overlap found between adjacent chunks: {chunks}"


def test_overlap_zero_no_carryover():
    """chunk_overlap=0 must not duplicate content across chunks."""
    s = TextSplitter(chunk_size=12, chunk_overlap=0)
    text = "".join(f"句{i}。" for i in range(1, 20))
    chunks = s.split(text)
    # With no overlap, the start of chunk[i+1] should not equal the tail of chunk[i].
    for i in range(len(chunks) - 1):
        tail = chunks[i][-4:]
        if tail:
            assert not chunks[i + 1].startswith(tail)


def test_no_empty_chunks():
    s = TextSplitter(chunk_size=32, chunk_overlap=8)
    text = "\n\n第一段内容。\n\n\n第二段内容。\n\n"
    chunks = s.split(text)
    assert all(c.strip() for c in chunks)


def test_force_split_when_no_separator():
    """A single long token with no separators is force-split by character."""
    s = TextSplitter(chunk_size=10, chunk_overlap=3)
    text = "a" * 50
    chunks = s.split(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 10


def test_split_documents_preserves_metadata():
    from rag_system.documents import Document

    s = TextSplitter(chunk_size=20, chunk_overlap=0)
    doc = Document(path="/tmp/x.md", content="句一。句二。句三。", metadata={"filename": "x.md"})
    chunks = s.split_documents([doc])
    assert len(chunks) >= 1
    for i, c in enumerate(chunks):
        assert c.source == "/tmp/x.md"
        assert c.chunk_index == i
        assert c.metadata.get("filename") == "x.md"
