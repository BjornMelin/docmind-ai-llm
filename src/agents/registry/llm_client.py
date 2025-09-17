"""Retry-aware shared LLM client utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


class RetryLLMClient:
    """Thin retry wrapper around a LangChain ``BaseLanguageModel`` instance.

    The coordinator relies on a shared LLM client to guarantee consistent retry
    behaviour, deadline awareness, and to provide a single place to hook in
    telemetry. This wrapper keeps the semantics of the underlying LLM while
    applying Tenacity's exponential backoff with jitter whenever a transient
    failure occurs. Only synchronous and asynchronous ``invoke``/``predict``
    calls are retried; streaming operations degrade to a simple passthrough to
    avoid replaying partial responses.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        *,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 8.0,
        retryable_exceptions: tuple[type[BaseException], ...] = (
            TimeoutError,
            OSError,
        ),
    ) -> None:
        """Instantiate the retry wrapper.

        Args:
            llm: Underlying language model instance.
            max_attempts: Maximum attempts before giving up (must be >= 1).
            initial_backoff_seconds: Base backoff value for the exponential
                jittered strategy.
            max_backoff_seconds: Upper bound for the backoff delay.
            retryable_exceptions: Exceptions that should trigger a retry.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._llm = llm
        self._max_attempts = max_attempts
        self._wait = wait_random_exponential(
            multiplier=initial_backoff_seconds, max=max_backoff_seconds
        )
        self._retry_condition = retry_if_exception_type(retryable_exceptions)

    @property
    def inner(self) -> BaseLanguageModel:
        """Return the wrapped language model."""
        return self._llm

    def _sync_retry(self) -> Retrying:
        """Build a Tenacity retry controller for sync operations."""
        return Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=self._retry_condition,
            reraise=True,
        )

    def _async_retry(self) -> AsyncRetrying:
        """Build a Tenacity retry controller for async operations."""
        return AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=self._retry_condition,
            reraise=True,
        )

    def _update_retry_state(self, state: RetryCallState) -> None:
        """Hook for future telemetry instrumentation.

        ``RetryCallState`` carries timing information for an attempt. The
        default implementation is a no-op but provides a single location to
        inject tracing in follow-up phases without modifying call sites.
        """

    def invoke(self, payload: Any, **kwargs: Any) -> Any:
        """Invoke the underlying LLM with retry semantics."""
        for attempt in self._sync_retry():
            with attempt:
                result = self._llm.invoke(payload, **kwargs)
            self._update_retry_state(attempt)
            return result
        raise RuntimeError("Retry loop failed to return a result")

    async def ainvoke(self, payload: Any, **kwargs: Any) -> Any:
        """Asynchronously invoke the underlying LLM with retry semantics."""
        async for attempt in self._async_retry():
            with attempt:
                result = await self._llm.ainvoke(payload, **kwargs)
            self._update_retry_state(attempt)
            return result
        raise RuntimeError("Async retry loop failed to return a result")

    def predict(self, text: str, **kwargs: Any) -> str:
        """Predict helper that honours retry semantics."""
        return self.invoke(text, **kwargs)

    async def apredict(self, text: str, **kwargs: Any) -> str:
        """Async variant of :meth:`predict`."""
        return await self.ainvoke(text, **kwargs)

    def stream(self, payload: Any, **kwargs: Any) -> Iterable[Any]:
        """Yield tokens by delegating to the wrapped LLM.

        Streaming is not retried to avoid duplicating partial responses. The
        calling code remains responsible for handling stream failures.
        """
        return self._llm.stream(payload, **kwargs)

    async def astream(self, payload: Any, **kwargs: Any) -> Any:
        """Async streaming passthrough to the wrapped LLM."""
        return await self._llm.astream(payload, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped LLM."""
        return getattr(self._llm, name)
