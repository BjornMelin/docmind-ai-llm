"""Retry-aware wrapper for LlamaIndex LLM instances.

LlamaIndex LLMs (e.g. ``Settings.llm``) expose ``complete``/``chat`` and async
variants. When the coordinator is configured to share a single LlamaIndex LLM,
we wrap it with retry semantics to preserve reliability for DSPy optimization
and any LlamaIndex-side calls that use the shared LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

try:  # pragma: no cover - typing-only imports
    from llama_index.core.base.llms.types import (  # type: ignore
        ChatMessage,
    )
except Exception:  # pragma: no cover - defensive for optional llama-index extras
    ChatMessage = Any  # type: ignore[assignment]


class RetryLlamaIndexLLM:
    """Thin retry wrapper around a LlamaIndex LLM instance.

    The wrapper retries non-streaming operations (``complete``/``chat`` and their
    async counterparts) with exponential backoff and jitter. Streaming calls are
    delegated directly to avoid replaying partial responses.
    """

    def __init__(
        self,
        llm: Any,
        *,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 8.0,
        retryable_exceptions: tuple[type[BaseException], ...] = (
            TimeoutError,
            ConnectionError,
        ),
    ) -> None:
        """Instantiate the retry wrapper.

        Args:
            llm: Underlying LlamaIndex LLM instance.
            max_attempts: Maximum attempts before giving up (must be >= 1).
            initial_backoff_seconds: Base backoff value for the exponential
                jittered strategy.
            max_backoff_seconds: Upper bound for the backoff delay.
            retryable_exceptions: Exceptions that should trigger a retry.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.inner = llm
        self._max_attempts = max_attempts
        self._wait = wait_random_exponential(
            multiplier=initial_backoff_seconds, max=max_backoff_seconds
        )
        self._retry_condition = retry_if_exception_type(retryable_exceptions)

    @property
    def metadata(self) -> Any:
        """Expose the wrapped LLM's metadata (if present)."""
        return getattr(self.inner, "metadata", None)

    def _sync_retry(self) -> Retrying:
        return Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=self._retry_condition,
            reraise=True,
        )

    def _async_retry(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=self._retry_condition,
            reraise=True,
        )

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        """Call ``inner.complete`` with retry semantics."""
        for attempt in self._sync_retry():
            with attempt:
                return self.inner.complete(prompt, **kwargs)
        raise RuntimeError("Retry loop failed to return a result")

    async def acomplete(self, prompt: str, **kwargs: Any) -> Any:
        """Call ``inner.acomplete`` with retry semantics."""
        async for attempt in self._async_retry():
            with attempt:
                return await self.inner.acomplete(prompt, **kwargs)
        raise RuntimeError("Async retry loop failed to return a result")

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        """Call ``inner.chat`` with retry semantics."""
        for attempt in self._sync_retry():
            with attempt:
                return self.inner.chat(messages, **kwargs)
        raise RuntimeError("Retry loop failed to return a result")

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        """Call ``inner.achat`` with retry semantics."""
        async for attempt in self._async_retry():
            with attempt:
                return await self.inner.achat(messages, **kwargs)
        raise RuntimeError("Async retry loop failed to return a result")

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        """Delegate to ``inner.stream_complete`` (no retries)."""
        return self.inner.stream_complete(prompt, **kwargs)

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        """Delegate to ``inner.stream_chat`` (no retries)."""
        return self.inner.stream_chat(messages, **kwargs)

    def astream_complete(self, prompt: str, **kwargs: Any) -> Any:
        """Delegate to ``inner.astream_complete`` (no retries)."""
        return self.inner.astream_complete(prompt, **kwargs)

    def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        """Delegate to ``inner.astream_chat`` (no retries)."""
        return self.inner.astream_chat(messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped LLM."""
        return getattr(self.inner, name)
