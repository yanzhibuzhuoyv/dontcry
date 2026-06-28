"""Recursive character text splitter with Chinese punctuation awareness.

Pure Python — no LangChain dependency. Same algorithm as
RecursiveCharacterTextSplitter but zero-dependency.
"""

from dataclasses import dataclass
from typing import Callable

from .config import ChunkingConfig
from .documents import Chunk, Document


# Separators ordered from highest priority to lowest.
# Chinese punctuation is prioritized above ASCII equivalents.
_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    "",
)


@dataclass(frozen=True)
class TextSplitter:
    """Splits text by recursively breaking on separator boundaries."""

    chunk_size: int = 512
    chunk_overlap: int = 128
    separators: tuple[str, ...] = _SEPARATORS
    length_function: Callable[[str], int] = len

    def split(self, text: str) -> list[str]:
        """Split a single text into chunks. Returns empty list for empty text."""
        if not text:
            return []
        chunks = self._split_text(text, list(self.separators))
        return [c for c in chunks if c.strip()]

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split each document, preserving source path metadata in each chunk."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            texts = self.split(doc.content)
            for i, text in enumerate(texts):
                all_chunks.append(
                    Chunk(
                        text=text,
                        source=doc.path,
                        chunk_index=i,
                        metadata=dict(doc.metadata),
                    )
                )
        return all_chunks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split: try first separator, recurse with next if chunks too big."""
        # Find first separator that exists in the text
        chosen_sep = ""
        chosen_idx = -1
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                chosen_sep = sep
                chosen_idx = i
                break
        else:
            # No separator matched — treat as char-split on empty string
            chosen_sep = ""
            chosen_idx = len(separators) - 1  # "" is last separator

        remaining = separators[chosen_idx + 1 :]

        if chosen_sep == "":
            pieces = list(text)
        else:
            pieces = text.split(chosen_sep)

        return self._merge_and_recurse(pieces, chosen_sep, remaining)

    def _merge_and_recurse(
        self,
        pieces: list[str],
        separator: str,
        remaining_separators: list[str],
    ) -> list[str]:
        """Merge pieces that fit in chunk_size; recurse on oversized ones."""
        final_chunks: list[str] = []
        good_splits: list[str] = []

        for piece in pieces:
            if self.length_function(piece) <= self.chunk_size:
                good_splits.append(piece)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                if remaining_separators:
                    final_chunks.extend(self._split_text(piece, remaining_separators))
                else:
                    # No more separators — force-split by character
                    stride = max(1, self.chunk_size - self.chunk_overlap)
                    for j in range(0, len(piece), stride):
                        sub = piece[j : j + self.chunk_size]
                        if sub.strip():
                            final_chunks.append(sub)

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge small adjacent splits into chunks that fit within chunk_size."""
        if not splits:
            return []

        sep_len = self.length_function(separator)
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for split in splits:
            split_len = self.length_function(split)
            add_sep_len = sep_len if current_chunk else 0

            if current_len + add_sep_len + split_len <= self.chunk_size:
                current_chunk.append(split)
                current_len += add_sep_len + split_len
            else:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                current_chunk = [split]
                current_len = split_len

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks


def create_splitter(config: ChunkingConfig) -> TextSplitter:
    """Factory: create a TextSplitter from ChunkingConfig."""
    return TextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
