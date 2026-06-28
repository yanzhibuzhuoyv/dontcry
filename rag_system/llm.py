"""OpenAI-compatible chat completion client with retry logic."""

import time
from dataclasses import dataclass

from .config import LLMGenerationConfig
from .errors import GenerationError
from .vector_store import SearchResult


@dataclass(frozen=True)
class _RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0


class LLMGenerator:
    """OpenAI-compatible synchronous chat completion client with retry."""

    def __init__(self, config: LLMGenerationConfig):
        from openai import OpenAI

        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=120.0,
        )
        self._retry = _RetryConfig()

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Send messages to chat completion, return response text.

        Raises GenerationError on failure after max retries.
        """
        last_error: Exception | None = None

        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise GenerationError("LLM returned empty response")
                return content
            except GenerationError:
                raise
            except Exception as exc:
                last_error = exc
                if not self._is_retryable(exc):
                    raise GenerationError(f"LLM generation failed: {exc}") from exc
                if attempt < self._retry.max_attempts:
                    delay = min(
                        self._retry.base_delay * (2 ** (attempt - 1)),
                        self._retry.max_delay,
                    )
                    time.sleep(delay)

        raise GenerationError(
            f"LLM generation failed after {self._retry.max_attempts} attempts: "
            f"{last_error}"
        )

    def build_rag_prompt(
        self,
        question: str,
        context_chunks: list[SearchResult],
    ) -> list[dict[str, str]]:
        """Construct messages with system prompt + context + question."""
        system_prompt = (
            "你是一个基于参考资料的问答助手。请根据以下参考资料回答用户的问题。\n"
            "规则：\n"
            "1. 如果参考资料包含答案，请基于资料内容回答，并在末尾注明引用的来源。\n"
            "2. 如果参考资料不包含答案，请明确说「参考资料中未找到相关信息」，不要编造。\n"
            "3. 回答简洁明了，直接回应问题。"
        )

        if context_chunks:
            context_text = "\n\n---\n\n".join(
                f"[来源: {c.source}]\n{c.text}" for c in context_chunks
            )
            system_prompt += f"\n\n## 参考资料\n\n{context_text}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True for transient errors (rate limits, server errors)."""
        status = _extract_status_code(exc)
        if status is not None:
            return status in (429, 500, 502, 503, 504)
        error_str = str(exc).lower()
        return any(
            kw in error_str
            for kw in ("timeout", "connection", "reset", "network", "retry")
        )


def _extract_status_code(exc: Exception) -> int | None:
    """Walk exception chain looking for an HTTP status code."""
    current: BaseException = exc
    while current is not None:
        for attr in ("status_code", "http_status", "status"):
            val = getattr(current, attr, None)
            if isinstance(val, int):
                return val
        if hasattr(current, "response"):
            resp = getattr(current, "response", None)
            if resp is not None:
                status = getattr(resp, "status_code", None)
                if isinstance(status, int):
                    return status
        current = current.__cause__ or current.__context__
    return None
