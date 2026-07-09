"""FAISS-backed local vector store with JSON metadata sidecar.

Uses FAISS IndexFlatIP (inner product = cosine similarity on normalized vectors).
Exact search — perfect recall, fine for <100K documents.
"""

import hashlib
import json
import os
import tempfile
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
        # source path → content_hash, for O(1) change detection
        self._source_hashes: dict[str, str] = {}

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

        # Normalise document vectors in-place so that inner product on this
        # IndexFlatIP equals cosine similarity regardless of whether the
        # embedder already normalised. Previously this was an implicit
        # contract with the embedder; making it explicit avoids silent
        # similarity distortion if a non-normalising embedder is plugged in.
        faiss = _get_faiss()
        faiss.normalize_L2(vectors)  # type: ignore[attr-defined]

        start_id = self._index.ntotal
        self._index.add(vectors)  # type: ignore[attr-defined]

        for i, chunk in enumerate(chunks):
            vid = start_id + i
            content_hash = chunk.metadata.get("content_hash", "")
            self._metadata[vid] = {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
                "content_hash": content_hash,
            }
            self._sources.add(chunk.source)
            if content_hash:
                self._source_hashes[chunk.source] = str(content_hash)

    def has_source(self, source: str) -> bool:
        """Return True if *source* has already been ingested."""
        return source in self._sources

    def file_hash(self, source: str) -> str | None:
        """Return the stored content hash for *source*, or None if not ingested."""
        # O(1) lookup via the source→hash index maintained in add/replace.
        return self._source_hashes.get(source)

    def remove_source(self, source: str) -> int:
        """Remove all chunks belonging to *source*.

        Returns the number of chunks removed (0 if the source was not
        present). Uses the same batch-reconstruct + rebuild approach as
        :meth:`replace_source` but adds nothing back.
        """
        _faiss = _get_faiss()
        remove_ids: set[int] = {
            vid for vid, meta in self._metadata.items()
            if meta.get("source") == source
        }
        if not remove_ids:
            return 0

        sorted_ids = sorted(self._metadata.keys())
        keep_mask = np.array(
            [vid not in remove_ids for vid in sorted_ids], dtype=bool
        )
        max_id = sorted_ids[-1]
        all_vectors = self._index.reconstruct_n(0, max_id + 1)  # type: ignore[attr-defined]
        keep_vectors = all_vectors[keep_mask]

        self._sources.discard(source)
        self._source_hashes.pop(source, None)
        for vid in remove_ids:
            del self._metadata[vid]

        new_index = _faiss.IndexFlatIP(self._dimension)
        if keep_vectors.shape[0] > 0:
            new_index.add(np.ascontiguousarray(keep_vectors))  # type: ignore[attr-defined]

        remaining = sorted(self._metadata.keys())
        new_metadata: dict[int, dict[str, object]] = {}
        for new_id, old_id in enumerate(remaining):
            new_metadata[new_id] = self._metadata[old_id]

        self._index = new_index
        self._metadata = new_metadata
        return len(remove_ids)

    def replace_source(
        self, source: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Replace chunks of *source* with new ones. Old entries are removed.

        Extract vectors directly from the FAISS index (IndexFlat stores them
        internally), filter out removed entries, rebuild. Uses batch
        ``reconstruct_n`` instead of per-position reconstruction for speed.
        """
        _faiss = _get_faiss()
        ntotal = self._index.ntotal

        # 1. Find which vector positions belong to *source*
        remove_ids: set[int] = {
            vid for vid, meta in self._metadata.items()
            if meta.get("source") == source
        }

        if ntotal == 0 or not remove_ids:
            self._sources.discard(source)
            self._source_hashes.pop(source, None)
            self.add_documents(chunks, embeddings)
            return

        # 2. Batch-extract all vectors from FAISS, then filter in NumPy.
        #    reconstruct_n(start, n) is far cheaper than n separate calls.
        sorted_ids = sorted(self._metadata.keys())
        keep_mask = np.array(
            [vid not in remove_ids for vid in sorted_ids], dtype=bool
        )
        if sorted_ids:
            # IndexFlat supports reconstruct_n over a contiguous range only,
            # so pull the whole span [0, max_id] once and select by mask.
            max_id = sorted_ids[-1]
            all_vectors = self._index.reconstruct_n(0, max_id + 1)  # type: ignore[attr-defined]
            keep_vectors = all_vectors[keep_mask]
        else:
            keep_vectors = np.empty((0, self._dimension), dtype=np.float32)

        # 3. Remove old entries
        self._sources.discard(source)
        self._source_hashes.pop(source, None)
        for vid in remove_ids:
            del self._metadata[vid]

        # 4. Rebuild FAISS index with kept vectors
        new_index = _faiss.IndexFlatIP(self._dimension)
        if keep_vectors.shape[0] > 0:
            new_index.add(np.ascontiguousarray(keep_vectors))  # type: ignore[attr-defined]

        # 5. Remap metadata keys (0, 1, 2, ...)
        remaining = sorted(self._metadata.keys())
        new_metadata: dict[int, dict[str, object]] = {}
        for new_id, old_id in enumerate(remaining):
            new_metadata[new_id] = self._metadata[old_id]

        self._index = new_index
        self._metadata = new_metadata

        # 6. Add new chunks
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
        """Persist index and metadata to disk atomically.

        Both files are written to temporary paths first and renamed into
        place only after both succeed. This prevents a crash between the two
        writes from leaving the store in an inconsistent (index ↔ metadata
        mismatched) state. Temp files are cleaned up on failure.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss = _get_faiss()
        index_path = directory / self.INDEX_FILENAME
        meta_path = directory / self.META_FILENAME

        serializable_meta = {str(k): v for k, v in self._metadata.items()}
        meta_payload = json.dumps(
            {
                "dimension": self._dimension,
                "sources": sorted(self._sources),
                "source_hashes": self._source_hashes,
                "entries": serializable_meta,
            },
            ensure_ascii=False,
            indent=2,
        )

        tmp_meta_path: Path | None = None
        tmp_index_path: Path | None = None
        try:
            # Write both temp files first; only rename after both succeed.
            tmp_fd, tmp_meta_name = tempfile.mkstemp(
                prefix=self.META_FILENAME + ".", suffix=".tmp", dir=str(directory)
            )
            os.close(tmp_fd)
            tmp_meta_path = Path(tmp_meta_name)
            tmp_meta_path.write_text(meta_payload, encoding="utf-8")

            tmp_index_path = directory / (self.INDEX_FILENAME + ".tmp")
            faiss.write_index(self._index, str(tmp_index_path))  # type: ignore[attr-defined]

            # Rename metadata first, then the index. If a crash happens
            # between the two renames, on the next load metadata will be a
            # superset of the index entries (a few orphan entries), which is
            # safe — search is bounded by index.ntotal. The reverse order
            # could let search return vector ids absent from metadata.
            os.replace(tmp_meta_path, meta_path)
            tmp_meta_path = None  # ownership transferred
            os.replace(tmp_index_path, index_path)
            tmp_index_path = None
        finally:
            # Clean up any temp files left over from a failed write.
            for leftover in (tmp_meta_path, tmp_index_path):
                if leftover is not None:
                    try:
                        leftover.unlink(missing_ok=True)
                    except OSError:
                        pass

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
        # Rebuild the source→hash index. Prefer the persisted map; fall back
        # to scanning entries for stores written by older versions.
        source_hashes = meta_json.get("source_hashes")
        if isinstance(source_hashes, dict):
            store._source_hashes = {str(k): str(v) for k, v in source_hashes.items()}
        else:
            store._source_hashes = {}
            for entry in store._metadata.values():
                src = entry.get("source")
                h = entry.get("content_hash")
                if src and h and src not in store._source_hashes:
                    store._source_hashes[str(src)] = str(h)
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
