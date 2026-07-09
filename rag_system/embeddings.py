"""Embedding providers: local sentence-transformers and API (OpenAI-compatible).

Both implement the Embedder Protocol so callers don't need to know which
backend is active.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .config import EmbeddingConfig
from .errors import EmbeddingError


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Callers program against this protocol, not concrete types."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# Local sentence-transformers embedder
# ---------------------------------------------------------------------------

# BGE models require this instruction prefix for optimal query embeddings
_BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


@dataclass(frozen=True)
class LocalEmbedder:
    """Loads a sentence-transformers model locally.

    The model is lazy-loaded on first embed call so that importing this
    module does not pull in PyTorch until needed.
    """

    model_name: str
    device: str = "cpu"
    _model: Any = field(default=None, repr=False, compare=False)
    _dim: int = field(default=0, repr=False, compare=False)

    def _ensure_loaded(self) -> None:
        """Lazy-load the SentenceTransformer on first embed call."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Install it: pip install sentence-transformers"
            )
        try:
            object.__setattr__(
                self, "_model",
                SentenceTransformer(self.model_name, device=self.device),
            )
            # Determine dimension from a tiny test encode
            dim = self._model.encode(["test"], normalize_embeddings=True).shape[1]
            object.__setattr__(self, "_dim", int(dim))
        except Exception as exc:
            raise EmbeddingError(
                f"failed to load local model '{self.model_name}': {exc}"
            ) from exc

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        try:
            vectors = self._model.encode(  # type: ignore[union-attr]
                texts,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 10,
                batch_size=32,
            )
            return [v.tolist() for v in vectors]
        except Exception as exc:
            raise EmbeddingError(f"local embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        self._ensure_loaded()
        query_text = (
            f"{_BGE_QUERY_PREFIX}{text}"
            if "bge" in self.model_name.lower()
            else text
        )
        try:
            vector = self._model.encode(  # type: ignore[union-attr]
                [query_text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vector[0].tolist()
        except Exception as exc:
            raise EmbeddingError(f"local query embedding failed: {exc}") from exc


# ---------------------------------------------------------------------------
# OpenAI-compatible API embedder
# ---------------------------------------------------------------------------

_API_BATCH_SIZE = 100
# Conservative max input length per text (chars). Most embedding models cap
# at 512-8192 tokens; ~16000 chars ≈ 8000 tokens for mixed CJK/ASCII. Longer
# inputs are truncated to avoid silent API rejection.
_API_MAX_INPUT_CHARS = 16000
_API_RETRY_ATTEMPTS = 3
_API_RETRY_BASE_DELAY = 1.0
_API_RETRY_MAX_DELAY = 30.0


def _truncate_text(text: str, max_chars: int = _API_MAX_INPUT_CHARS) -> str:
    """Truncate text to max_chars to stay under embedding model input limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _is_retryable_embedding_error(exc: Exception) -> bool:
    """Return True for transient embedding API errors (rate limit, server).

    Walks the exception chain to find an HTTP status code — the openai SDK
    often wraps the real status inside ``exc.response.status_code`` rather
    than on the exception itself. Mirrors the logic in ``llm._extract_status_code``
    so both clients agree on what is retryable.
    """
    status: int | None = None
    current: BaseException | None = exc
    while current is not None:
        for attr in ("status_code", "http_status", "status"):
            val = getattr(current, attr, None)
            if isinstance(val, int):
                status = val
                break
        if status is None:
            resp = getattr(current, "response", None)
            if resp is not None:
                val = getattr(resp, "status_code", None)
                if isinstance(val, int):
                    status = val
        if status is not None:
            break
        current = current.__cause__ or current.__context__
    if status is not None:
        return status in (429, 500, 502, 503, 504)
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in ("timeout", "connection", "reset", "network", "retry", "overloaded")
    )


@dataclass(frozen=True)
class APIEmbedder:
    """Uses openai.OpenAI client against any compatible endpoint."""

    model: str
    base_url: str
    api_key: str
    _client: Any = field(default=None, repr=False, compare=False)
    _dim: int = field(default=0, repr=False, compare=False)

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise EmbeddingError(
                "openai package is not installed. Install it: pip install openai"
            )
        object.__setattr__(
            self,
            "_client",
            OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=60.0),
        )

    @property
    def dimension(self) -> int:
        if self._dim == 0:
            self._ensure_client()
            try:
                resp = self._client.embeddings.create(  # type: ignore[union-attr]
                    model=self.model, input=["dimension probe"]
                )
                object.__setattr__(self, "_dim", len(resp.data[0].embedding))
            except Exception as exc:
                raise EmbeddingError(
                    f"failed to probe embedding dimension: {exc}"
                ) from exc
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_client()
        # Truncate inputs to stay under embedding model input limits.
        safe_texts = [_truncate_text(t) for t in texts]
        all_vectors: list[list[float]] = []
        for i in range(0, len(safe_texts), _API_BATCH_SIZE):
            batch = safe_texts[i : i + _API_BATCH_SIZE]
            batch_vectors = self._embed_batch_with_retry(
                batch, i // _API_BATCH_SIZE
            )
            all_vectors.extend(batch_vectors)
        return all_vectors

    def _embed_batch_with_retry(
        self, batch: list[str], batch_idx: int
    ) -> list[list[float]]:
        """Embed one batch with exponential-backoff retry on transient errors.

        Previously a single 429/503 would abort the whole ingest; now only
        non-retryable errors fail immediately, and transient ones are retried
        (mirroring the policy already used by LLMGenerator).
        """
        last_error: Exception | None = None
        for attempt in range(1, _API_RETRY_ATTEMPTS + 1):
            try:
                resp = self._client.embeddings.create(  # type: ignore[union-attr]
                    model=self.model, input=batch
                )
                items = sorted(resp.data, key=lambda d: d.index)
                return [it.embedding for it in items]
            except Exception as exc:
                last_error = exc
                if not _is_retryable_embedding_error(exc):
                    raise EmbeddingError(
                        f"API embedding failed (batch {batch_idx}): {exc}"
                    ) from exc
                if attempt < _API_RETRY_ATTEMPTS:
                    delay = min(
                        _API_RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                        _API_RETRY_MAX_DELAY,
                    )
                    time.sleep(delay)
        raise EmbeddingError(
            f"API embedding failed after {_API_RETRY_ATTEMPTS} attempts "
            f"(batch {batch_idx}): {last_error}"
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Factory: returns LocalEmbedder or APIEmbedder based on config.provider."""
    if config.provider == "local":
        return LocalEmbedder(model_name=config.model, device=config.device)
    return APIEmbedder(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
    )
