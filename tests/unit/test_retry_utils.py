"""Focused tests for utils/retry_utils.py - testing only actual exported functions.

Tests the tenacity-based retry utilities with focus on what's actually implemented.
Target coverage: 80%+ for retry utilities.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

# Mock the logging dependencies early
sys.modules["utils.logging_config"] = MagicMock()
logger_mock = MagicMock()
sys.modules["utils.logging_config"].logger = logger_mock

# Mock exceptions module to avoid the logging chain
sys.modules["utils.exceptions"] = MagicMock()
sys.modules["utils.model_manager"] = MagicMock()

from utils.retry_utils import (  # noqa: E402
    async_managed_resource,
    async_with_timeout,
    document_retry,
    embedding_retry,
    llm_retry,
    managed_resource,
    safe_execute,
    safe_execute_async,
    standard_retry,
    with_fallback,
)


class TestRetryDecorators:
    """Test suite for tenacity-based retry decorators."""

    def test_standard_retry_success(self):
        """Test standard_retry with successful function."""
        call_count = 0

        @standard_retry
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()
        assert result == "success"
        assert call_count == 1

    def test_embedding_retry_with_retries(self):
        """Test embedding_retry with failures then success."""
        call_count = 0

        @embedding_retry
        def flaky_embedding():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "embedding_result"

        result = flaky_embedding()
        assert result == "embedding_result"
        assert call_count == 3

    def test_llm_retry_exhausted(self):
        """Test llm_retry when all retries are exhausted."""
        call_count = 0

        @llm_retry
        def always_failing_llm():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            always_failing_llm()

        # Should have tried 5 times (stop_after_attempt(5))
        assert call_count == 5


class TestWithFallback:
    """Test suite for with_fallback decorator."""

    def test_with_fallback_primary_success(self):
        """Test fallback when primary function succeeds."""
        fallback_called = False

        def fallback_function():
            nonlocal fallback_called
            fallback_called = True
            return "fallback"

        @with_fallback(fallback_function)
        def primary_function():
            return "primary"

        result = primary_function()

        assert result == "primary"
        assert not fallback_called

    def test_with_fallback_primary_fails(self):
        """Test fallback when primary function fails."""
        logger_mock.reset_mock()

        def fallback_function():
            return "fallback_success"

        @with_fallback(fallback_function)
        def failing_primary():
            raise ConnectionError("Primary failed")

        result = failing_primary()

        assert result == "fallback_success"

    def test_with_fallback_both_fail(self):
        """Test when both primary and fallback functions fail."""
        logger_mock.reset_mock()

        def failing_fallback():
            raise RuntimeError("Fallback also failed")

        @with_fallback(failing_fallback)
        def failing_primary():
            raise ConnectionError("Primary failed")

        with pytest.raises(RuntimeError, match="Fallback also failed"):
            failing_primary()


class TestAsyncWithTimeout:
    """Test suite for async_with_timeout decorator."""

    @pytest.mark.asyncio
    async def test_async_with_timeout_success(self):
        """Test async function completes within timeout."""

        @async_with_timeout(timeout_seconds=1.0)
        async def quick_async_function():
            await asyncio.sleep(0.1)
            return "completed"

        result = await quick_async_function()
        assert result == "completed"

    @pytest.mark.asyncio
    async def test_async_with_timeout_exceeds(self):
        """Test async function exceeds timeout."""

        @async_with_timeout(timeout_seconds=0.1)
        async def slow_async_function():
            await asyncio.sleep(0.5)
            return "completed"

        with pytest.raises(TimeoutError, match="exceeded 0.1s timeout"):
            await slow_async_function()


class TestManagedResource:
    """Test suite for managed_resource context manager."""

    def test_managed_resource_success(self):
        """Test managed resource with successful operation."""
        resource = MagicMock()
        factory = Mock(return_value=resource)

        with managed_resource(factory) as managed:
            assert managed is resource

        factory.assert_called_once()
        resource.close.assert_called_once()

    def test_managed_resource_with_exception(self):
        """Test managed resource cleanup with exception."""
        resource = MagicMock()
        factory = Mock(return_value=resource)

        with (
            pytest.raises(ValueError, match="Test exception"),
            managed_resource(factory) as _,
        ):
            raise ValueError("Test exception")

        resource.close.assert_called_once()

    def test_managed_resource_custom_cleanup(self):
        """Test managed resource with custom cleanup function."""
        resource = MagicMock()
        factory = Mock(return_value=resource)
        cleanup = Mock()

        with managed_resource(factory, cleanup_func=cleanup):
            pass

        cleanup.assert_called_once_with(resource)
        resource.close.assert_not_called()  # Custom cleanup used instead


class TestAsyncManagedResource:
    """Test suite for async_managed_resource context manager."""

    @pytest.mark.asyncio
    async def test_async_managed_resource_success(self):
        """Test async managed resource with successful operation."""
        resource = AsyncMock()
        factory = AsyncMock(return_value=resource)

        async with async_managed_resource(factory) as managed:
            assert managed is resource

        factory.assert_called_once()
        resource.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_managed_resource_sync_factory(self):
        """Test async managed resource with sync factory."""
        resource = AsyncMock()
        factory = Mock(return_value=resource)  # Sync factory

        async with async_managed_resource(factory) as managed:
            assert managed is resource

        factory.assert_called_once()


class TestSafeExecute:
    """Test suite for safe_execute function."""

    def test_safe_execute_success(self):
        """Test safe_execute with successful function."""

        def successful_function():
            return "success"

        result = safe_execute(successful_function)
        assert result == "success"

    def test_safe_execute_with_exception(self):
        """Test safe_execute with exception."""
        logger_mock.reset_mock()

        def failing_function():
            raise ValueError("Function failed")

        result = safe_execute(failing_function, default_value="default")

        assert result == "default"

    def test_safe_execute_no_logging(self):
        """Test safe_execute with logging disabled."""

        def failing_function():
            raise RuntimeError("Function failed")

        result = safe_execute(
            failing_function, log_errors=False, default_value="quiet_default"
        )

        assert result == "quiet_default"


class TestSafeExecuteAsync:
    """Test suite for safe_execute_async function."""

    @pytest.mark.asyncio
    async def test_safe_execute_async_success(self):
        """Test safe_execute_async with successful function."""

        async def successful_async():
            return "async_success"

        result = await safe_execute_async(successful_async)
        assert result == "async_success"

    @pytest.mark.asyncio
    async def test_safe_execute_async_with_exception(self):
        """Test safe_execute_async with exception."""
        logger_mock.reset_mock()

        async def failing_async():
            raise ConnectionError("Async function failed")

        result = await safe_execute_async(failing_async, default_value="async_default")

        assert result == "async_default"

    @pytest.mark.asyncio
    async def test_safe_execute_async_timeout(self):
        """Test safe_execute_async with timeout."""

        async def slow_async():
            await asyncio.sleep(1.0)
            return "slow_result"

        result = await safe_execute_async(
            slow_async, timeout_seconds=0.1, default_value="timeout_handled"
        )

        assert result == "timeout_handled"


class TestRetryIntegration:
    """Integration tests for retry patterns working together."""

    def test_combined_decorators(self):
        """Test combining fallback with retry decorator."""
        call_count = 0

        @with_fallback(lambda: "fallback_result")
        @document_retry  # This will retry, then fallback will catch final failure
        def combined_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        result = combined_function()
        assert result == "fallback_result"
        # Should have tried document_retry attempts (3) then fallback
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_timeout_with_safe_execute(self):
        """Test async timeout combined with safe execution."""

        @async_with_timeout(timeout_seconds=1.0)
        async def async_function():
            await asyncio.sleep(0.1)
            return "async_success"

        result = await safe_execute_async(async_function, default_value="safe_default")
        assert result == "async_success"

        # Test timeout case
        @async_with_timeout(timeout_seconds=0.05)
        async def slow_function():
            await asyncio.sleep(0.2)
            return "slow_success"

        result = await safe_execute_async(slow_function, default_value="timeout_safe")
        assert result == "timeout_safe"
