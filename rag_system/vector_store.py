"""FAISS-backed local vector store with JSON metadata sidecar.

Uses FAISS IndexFlatIP (inner product = cosine similarity on normalized vectors).
Exact search — perfect recall, fine for <100K documents.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from .documents import Chunk
from .errors import VectorStoreError


@dataclass(frozen=True)
class SearchResult:
    """Single retrieval result."""

    text: str
    score: float
    source: str
    metadata: dict[str, str]


class VectorStore:
    """FAISS-backed vector store with JSON metadata sidecar.

    Directory layout::

        <store_dir>/
            index.faiss       # FAISS binary index
            metadata.json     # {dimension, sources, entries: {vector_id: {...}}}
    """

    INDEX_FILENAME: ClassVar[str] = "index.faiss"
    META_FILENAME: ClassVar[str] = "metadata.json"

    def __init__(self, dimension: int):
        """Initialize empty FAISS index.

        *dimension*: embedding vector dimension (e.g., 1536 for text-embedding-3-small).
        """
        faiss = _get_faiss()
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        # Map: vector_id (int) → metadata dict
        self._metadata: dict[int, dict[str, object]] = {}
        # Set of ingested source paths for idempotency
        self._sources: set[str] = set()

    @property
    def count(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal  # type: ignore[no-any-return]

    @property
    def dimension(self) -> int:
        return self._dimension

    def add_documents(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Add chunks with their pre-computed embeddings to the index.

        Raises VectorStoreError if lengths don't match.
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
            )
        if not chunks:
            return

        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.shape[1] != self._dimension:
            raise VectorStoreError(
                f"embedding dimension mismatch: expected {self._dimension}, "
                f"got {vectors.shape[1]}"
            )

        start_id = self._index.ntotal
        self._index.add(vectors)  # type: ignore[attr-defined]

        for i, chunk in enumerate(chunks):
            vid = start_id + i
            self._metadata[vid] = {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
                "content_hash": chunk.metadata.get("content_hash", ""),
            }
            self._sources.add(chunk.source)

    def has_source(self, source: str) -> bool:
        """Return True if *source* has already been ingested."""
        return source in self._sources

    def file_hash(self, source: str) -> str | None:
        """Return the stored content hash for *source*, or None if not ingested."""
        if source not in self._sources:
            return None
        # Find any entry with this source and return its stored hash
        for meta in self._metadata.values():
            if meta.get("source") == source:
                h = meta.get("content_hash")
                return str(h) if h else None
        return None

    def replace_source(
        self, source: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Replace chunks of *source* with new ones. Old entries are removed."""
        _faiss = _get_faiss()
        # Remove old entries for this source
        self._sources.discard(source)
        old_ids = [
            vid for vid, meta in self._metadata.items()
            if meta.get("source") == source
        ]
        for vid in old_ids:
            del self._metadata[vid]
        # Rebuild FAISS index without removed entries (IndexFlat has no remove)
        if self._metadata:
            remaining_vectors = np.array(
                [self._metadata[vid]["_vec"] for vid in sorted(self._metadata.keys())],
                dtype=np.float32,
            )
            new_index = _faiss.IndexFlatIP(self._dimension)
            new_index.add(remaining_vectors)  # type: ignore[attr-defined]
            self._index = new_index
            # Remap metadata keys
            sorted_ids = sorted(self._metadata.keys())
            new_metadata: dict[int, dict[str, object]] = {}
            for new_id, old_id in enumerate(sorted_ids):
                new_metadata[new_id] = self._metadata[old_id]
            self._metadata = new_metadata
        else:
            self._index = _faiss.IndexFlatIP(self._dimension)
        # Add new chunks
        self.add_documents(chunks, embeddings)

    def search(
        self, query_embedding: list[float], k: int = 5
    ) -> list[SearchResult]:
        """Return top-k most similar chunks by cosine similarity."""
        if self._index.ntotal == 0:
            return []

        faiss = _get_faiss()
        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)  # type: ignore[attr-defined]

        scores, indices = self._index.search(q, min(k, self._index.ntotal))  # type: ignore[attr-defined]

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx not in self._metadata:
                continue
            meta = self._metadata[idx]
            results.append(
                SearchResult(
                    text=str(meta["text"]),
                    score=float(score),
                    source=str(meta["source"]),
                    metadata={
                        str(k): str(v)
                        for k, v in meta.get("metadata", {}).items()  # type: ignore[arg-type]
                    },
                )
            )
        return results

    def save(self, directory: str | Path) -> None:
        """Persist index and metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss = _get_faiss()
        index_path = directory / self.INDEX_FILENAME
        faiss.write_index(self._index, str(index_path))  # type: ignore[attr-defined]

        serializable_meta = {str(k): v for k, v in self._metadata.items()}
        meta_path = directory / self.META_FILENAME
        meta_path.write_text(
            json.dumps(
                {
                    "dimension": self._dimension,
                    "sources": sorted(self._sources),
                    "entries": serializable_meta,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Load index and metadata from disk.

        Raises VectorStoreError if files are missing or corrupt.
        """
        directory = Path(directory)
        index_path = directory / cls.INDEX_FILENAME
        meta_path = directory / cls.META_FILENAME

        if not index_path.exists():
            raise VectorStoreError(f"index file not found: {index_path}")
        if not meta_path.exists():
            raise VectorStoreError(f"metadata file not found: {meta_path}")

        faiss = _get_faiss()
        index = faiss.read_index(str(index_path))  # type: ignore[attr-defined]

        meta_json = json.loads(meta_path.read_text(encoding="utf-8"))

        store = cls.__new__(cls)
        store._dimension = int(meta_json["dimension"])
        store._index = index
        store._metadata = {int(k): v for k, v in meta_json["entries"].items()}
        store._sources = set(meta_json.get("sources", []))
        return store

    @property
    def sources(self) -> set[str]:
        """Return a copy of the ingested source paths."""
        return set(self._sources)


def _get_faiss():
    """Lazy faiss import with a friendly error message."""
    try:
        import faiss  # type: ignore[import-untyped]

        return faiss
    except ImportError:
        raise VectorStoreError(
            "faiss-cpu is not installed. Install it: pip install faiss-cpu"
        )
