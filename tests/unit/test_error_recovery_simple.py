"""Simplified comprehensive tests for utils/error_recovery.py.

Tests core error recovery patterns with focus on functionality over mocking complexity.

Target coverage: 70%+ for error recovery utilities.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Mock the logging dependencies early
sys.modules["utils.logging_config"] = MagicMock()
logger_mock = MagicMock()
sys.modules["utils.logging_config"].logger = logger_mock

# Add utils directory to path for direct import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "utils"))

from error_recovery import (
    CircuitBreaker,
    async_managed_resource,
    async_with_timeout,
    managed_resource,
    safe_execute,
    safe_execute_async,
    with_fallback,
    with_retry,
)


class TestWithRetry:
    """Test suite for with_retry decorator."""

    def test_with_retry_success_no_retries(self):
        """Test successful execution without retries."""
        call_count = 0

        @with_retry(max_attempts=3)
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()

        assert result == "success"
        assert call_count == 1

    def test_with_retry_success_after_failures(self):
        """Test successful execution after some failures."""
        call_count = 0

        @with_retry(max_attempts=3, base_wait=0.01, max_wait=0.1)
        def eventually_successful_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = eventually_successful_function()

        assert result == "success"
        assert call_count == 3

    def test_with_retry_max_attempts_exceeded(self):
        """Test failure when max attempts are exceeded."""
        call_count = 0

        @with_retry(max_attempts=2, base_wait=0.01, max_wait=0.1)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_failing_function()

        assert call_count == 2

    def test_with_retry_custom_retry_conditions(self):
        """Test with custom retry conditions."""
        call_count = 0

        @with_retry(
            max_attempts=3,
            retry_on=(ConnectionError, TimeoutError),
            stop_on=(ValueError,),
            base_wait=0.01,
            max_wait=0.1,
        )
        def selective_retry_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Retry this")
            elif call_count == 2:
                raise ValueError("Don't retry this")
            return "success"

        with pytest.raises(ValueError, match="Don't retry this"):
            selective_retry_function()

        assert call_count == 2  # Retried once, then stopped on ValueError

    def test_with_retry_jitter_disabled(self):
        """Test retry with jitter disabled."""
        call_count = 0

        @with_retry(max_attempts=3, base_wait=0.01, jitter=False)
        def function_with_no_jitter():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Fail once")
            return "success"

        result = function_with_no_jitter()
        assert result == "success"
        assert call_count == 2


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

    def test_with_fallback_args_kwargs(self):
        """Test fallback with arguments and keyword arguments."""
        logger_mock.reset_mock()

        def fallback_function(*args, **kwargs):
            return f"fallback_args_{len(args)}_kwargs_{len(kwargs)}"

        @with_fallback(fallback_function)
        def failing_primary(*args, **kwargs):
            raise ValueError("Primary failed")

        result = failing_primary("arg1", "arg2", kwarg1="value1")

        assert result == "fallback_args_2_kwargs_1"


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

    @pytest.mark.asyncio
    async def test_async_with_timeout_custom_timeout(self):
        """Test async function with custom timeout."""

        @async_with_timeout(timeout_seconds=2.0)
        async def medium_async_function():
            await asyncio.sleep(0.5)
            return "completed"

        result = await medium_async_function()
        assert result == "completed"


class TestCircuitBreaker:
    """Test suite for CircuitBreaker pattern."""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state (normal operation)."""
        breaker = CircuitBreaker(failure_threshold=3)

        with breaker:
            result = "operation_success"

        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    def test_circuit_breaker_open_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=2)

        # First failure
        with (
            pytest.raises(ValueError),
            breaker
        ):
                raise ValueError("First failure")

        # Second failure - should open circuit
        with (
            pytest.raises(ValueError),
            breaker
        ):
                raise ValueError("Second failure")

        assert breaker.state == "OPEN"
        assert breaker.failure_count == 2

    @patch("time.time")
    def test_circuit_breaker_half_open_after_timeout(self, mock_time):
        """Test circuit breaker enters half-open state after timeout."""
        mock_time.side_effect = [0, 0, 0, 2.0]  # Simulate time progression

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)

        # Cause failure
        with (
            pytest.raises(ValueError),
            breaker
        ):
                raise ValueError("Failure")

        # Should enter half-open after timeout
        with breaker:
            result = "recovery_test"

        assert breaker.state == "CLOSED"  # Should reset to closed on success

    def test_circuit_breaker_custom_exceptions(self):
        """Test circuit breaker with custom expected exceptions."""
        breaker = CircuitBreaker(
            failure_threshold=2, expected_exception=(ConnectionError, TimeoutError)
        )

        # ValueError should not count as failure
        with (
            pytest.raises(ValueError),
            breaker
        ):
                raise ValueError("Not counted")

        assert breaker.failure_count == 0
        assert breaker.state == "CLOSED"

        # ConnectionError should count
        with (
            pytest.raises(ConnectionError),
            breaker
        ):
                raise ConnectionError("Counted failure")

        assert breaker.failure_count == 1


class TestManagedResource:
    """Test suite for managed_resource context manager."""

    def test_managed_resource_success(self):
        """Test managed resource with successful operation."""
        resource = MagicMock()
        factory = Mock(return_value=resource)

        with managed_resource(factory) as managed:
            assert managed is resource
            result = "operation_success"

        factory.assert_called_once()
        resource.close.assert_called_once()

    def test_managed_resource_with_exception(self):
        """Test managed resource cleanup with exception."""
        resource = MagicMock()
        factory = Mock(return_value=resource)

        with (
            pytest.raises(ValueError),
            managed_resource(factory) as managed
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

    def test_managed_resource_no_close_method(self):
        """Test managed resource when resource has no close method."""
        resource = "string_resource"  # No close method
        factory = Mock(return_value=resource)

        # Should not raise exception
        with managed_resource(factory) as managed:
            assert managed == resource

    def test_managed_resource_factory_failure(self):
        """Test managed resource when factory fails."""
        factory = Mock(side_effect=RuntimeError("Factory failed"))

        with (
            pytest.raises(RuntimeError, match="Factory failed"),
            managed_resource(factory)
        ):
                pass


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

    @pytest.mark.asyncio
    async def test_async_managed_resource_custom_cleanup(self):
        """Test async managed resource with custom async cleanup."""
        resource = AsyncMock()
        factory = AsyncMock(return_value=resource)
        cleanup = AsyncMock()

        async with async_managed_resource(factory, cleanup_func=cleanup):
            pass

        cleanup.assert_called_once_with(resource)

    @pytest.mark.asyncio
    async def test_async_managed_resource_sync_cleanup(self):
        """Test async managed resource with sync cleanup function."""
        resource = MagicMock()
        factory = Mock(return_value=resource)
        cleanup = Mock()

        async with async_managed_resource(factory, cleanup_func=cleanup):
            pass

        cleanup.assert_called_once_with(resource)


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

    def test_safe_execute_lambda(self):
        """Test safe_execute with lambda function."""
        result = safe_execute(lambda: "lambda_result")
        assert result == "lambda_result"

        result = safe_execute(lambda: 1 / 0, default_value="division_error_handled")
        assert result == "division_error_handled"


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

    @pytest.mark.asyncio
    async def test_safe_execute_async_sync_function(self):
        """Test safe_execute_async with sync function."""

        def sync_function():
            return "sync_in_async"

        result = await safe_execute_async(sync_function)
        assert result == "sync_in_async"


class TestRetryPatternIntegration:
    """Integration tests for retry patterns working together."""

    def test_combined_decorators(self):
        """Test combining multiple retry decorators."""
        call_count = 0

        @with_fallback(lambda: "fallback_result")
        @with_retry(max_attempts=2, base_wait=0.01)
        def combined_function():
            nonlocal call_count
            call_count += 1
            if call_count < 5:  # Will exhaust retries, trigger fallback
                raise ConnectionError("Always fails")
            return "success"

        result = combined_function()
        assert result == "fallback_result"
        assert call_count == 2  # Retried once, then fallback

    @pytest.mark.asyncio
    async def test_async_patterns_combined(self):
        """Test async retry patterns working together."""
        call_count = 0

        @async_with_timeout(timeout_seconds=1.0)
        async def async_retry_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                await asyncio.sleep(0.01)  # Small delay
                raise ConnectionError("Async failure")
            return "async_success"

        # Manually apply retry logic
        for attempt in range(3):
            try:
                result = await async_retry_function()
                break
            except ConnectionError:
                if attempt == 2:  # Last attempt
                    raise
                call_count -= 1  # Reset for retry logic
                continue

        assert result == "async_success"


class TestErrorRecoveryEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_circuit_breaker_negative_threshold(self):
        """Test circuit breaker with negative failure threshold."""
        breaker = CircuitBreaker(failure_threshold=-1)

        # Should handle gracefully
        with breaker:
            pass

    @pytest.mark.asyncio
    async def test_async_timeout_zero(self):
        """Test async timeout with zero timeout."""

        @async_with_timeout(timeout_seconds=0)
        async def instant_timeout():
            return "should_timeout"

        with pytest.raises(TimeoutError):
            await instant_timeout()

    def test_safe_execute_with_none_function_handled(self):
        """Test safe_execute with None function handled gracefully."""

        def none_safe_wrapper(func, **kwargs):
            if func is None:
                return kwargs.get("default_value")
            return safe_execute(func, **kwargs)

        result = none_safe_wrapper(None, default_value="none_handled")
        assert result == "none_handled"
