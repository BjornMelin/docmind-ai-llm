"""Unit tests for RetryLlamaIndexLLM wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.registry.llamaindex_llm_client import RetryLlamaIndexLLM


def test_complete_retries_on_timeouterror() -> None:
    """complete() retries transient failures."""
    inner = Mock()
    inner.complete = Mock(side_effect=[TimeoutError("t"), "ok"])

    wrapper = RetryLlamaIndexLLM(
        inner,
        max_attempts=2,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
    )

    assert wrapper.complete("prompt") == "ok"
    assert inner.complete.call_count == 2


def test_complete_retries_on_connectionerror() -> None:
    """complete() retries transient failures (ConnectionError)."""
    inner = Mock()
    inner.complete = Mock(side_effect=[ConnectionError("c"), "ok"])

    wrapper = RetryLlamaIndexLLM(
        inner,
        max_attempts=2,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
    )

    assert wrapper.complete("prompt") == "ok"
    assert inner.complete.call_count == 2


@pytest.mark.asyncio
async def test_acomplete_retries_on_timeouterror() -> None:
    """acomplete() retries transient failures."""
    inner = Mock()
    inner.acomplete = AsyncMock(side_effect=[TimeoutError("t"), "ok"])

    wrapper = RetryLlamaIndexLLM(
        inner,
        max_attempts=2,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
    )

    assert await wrapper.acomplete("prompt") == "ok"
    assert inner.acomplete.call_count == 2


@pytest.mark.asyncio
async def test_acomplete_retries_on_connectionerror() -> None:
    """acomplete() retries transient failures (ConnectionError)."""
    inner = Mock()
    inner.acomplete = AsyncMock(side_effect=[ConnectionError("c"), "ok"])

    wrapper = RetryLlamaIndexLLM(
        inner,
        max_attempts=2,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
    )

    assert await wrapper.acomplete("prompt") == "ok"
    assert inner.acomplete.call_count == 2


def test_chat_retries_on_connectionerror() -> None:
    """chat() retries transient failures (ConnectionError)."""
    inner = Mock()
    inner.chat = Mock(side_effect=[ConnectionError("c"), "ok"])

    wrapper = RetryLlamaIndexLLM(
        inner,
        max_attempts=2,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
    )

    assert wrapper.chat([Mock()]) == "ok"
    assert inner.chat.call_count == 2


@pytest.mark.asyncio
async def test_achat_retries_on_connectionerror() -> None:
    """achat() retries transient failures (ConnectionError)."""
    inner = Mock()
    inner.achat = AsyncMock(side_effect=[ConnectionError("c"), "ok"])

    wrapper = RetryLlamaIndexLLM(
        inner,
        max_attempts=2,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
    )

    assert await wrapper.achat([Mock()]) == "ok"
    assert inner.achat.call_count == 2


def test_wrapper_validates_max_attempts() -> None:
    """RetryLlamaIndexLLM rejects max_attempts < 1."""
    with pytest.raises(ValueError, match="max_attempts"):
        RetryLlamaIndexLLM(Mock(), max_attempts=0)


def test_complete_does_not_retry_non_transient_error() -> None:
    """complete() does not retry non-retryable exceptions."""
    inner = Mock()
    inner.complete = Mock(side_effect=ValueError("nope"))

    wrapper = RetryLlamaIndexLLM(
        inner,
        max_attempts=3,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
    )

    with pytest.raises(ValueError, match="nope"):
        wrapper.complete("prompt")
    assert inner.complete.call_count == 1
