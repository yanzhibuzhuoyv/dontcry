"""Lightweight Okapi BM25 implementation (zero dependencies).

Used for hybrid retrieval: BM25 catches exact keyword / rare-term matches
that dense vector search can miss. For Chinese text we index character
bigrams (no external tokenizer needed), which gives usable keyword
granularity without pulling in jieba — two-character terms dominate Chinese
vocabulary and bigrams capture most of them.
"""

import math
from collections import Counter
from typing import Sequence


def _tokenize(text: str) -> list[str]:
    """Tokenize for BM25: ASCII words (lowercased) + CJK character bigrams.

    Individual Han chars are too granular for keyword matching; bigrams of
    adjacent Han chars capture the bulk of two-character Chinese terms.
    """
    tokens: list[str] = []
    current_ascii: list[str] = []
    prev_han: str | None = None

    def _flush_ascii() -> None:
        if current_ascii:
            tokens.append("".join(current_ascii).lower())
            current_ascii.clear()

    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            _flush_ascii()
            if prev_han is not None:
                tokens.append(prev_han + ch)
            prev_han = ch
        elif ch.isalnum():
            current_ascii.append(ch)
            prev_han = None
        else:
            _flush_ascii()
            prev_han = None
    _flush_ascii()
    return tokens


class BM25Index:
    """In-memory Okapi BM25 index over a corpus of documents.

    Built from a list of candidate texts (typically the vector-recall
    candidate set) and queried per user query. Rebuilt per retrieval over the
    candidate set — cheap because the candidate set is small (top_k * mult).
    """

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._doc_len: list[int] = []
        for doc in corpus:
            toks = _tokenize(doc)
            self._doc_tokens.append(toks)
            self._doc_len.append(len(toks))
            for term in set(toks):
                self._doc_freq[term] += 1
        self._n = len(self._doc_tokens)
        self._avgdl = (sum(self._doc_len) / self._n) if self._n else 0.0

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        # Okapi IDF with +1 smoothing to keep it non-negative.
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        """Return BM25 score for every document against the query."""
        q_terms = _tokenize(query)
        if not q_terms or self._n == 0:
            return [0.0] * self._n
        results = [0.0] * self._n
        avgdl = self._avgdl or 1.0
        for term in set(q_terms):
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for i, toks in enumerate(self._doc_tokens):
                tf = toks.count(term)
                if tf == 0:
                    continue
                dl = self._doc_len[i]
                denom = tf + self._k1 * (1 - self._b + self._b * dl / avgdl)
                results[i] += idf * (tf * (self._k1 + 1)) / denom
        return results
