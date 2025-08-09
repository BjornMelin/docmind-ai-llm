"""Comprehensive tests for utils/error_recovery.py.

Tests error recovery patterns with comprehensive coverage of retry decorators,
circuit breaker, timeout handling, and resource management utilities.

Target coverage: 95%+ for error recovery utilities.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Mock the logging and other dependencies before importing
sys.modules["utils.logging_config"] = MagicMock()
logger_mock = MagicMock()
sys.modules["utils.logging_config"].logger = logger_mock

# Mock exceptions module to avoid the logging chain
sys.modules["utils.exceptions"] = MagicMock()

# Add utils directory to path for direct import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "utils"))

from error_recovery import (
    CircuitBreaker,
    async_managed_resource,
    async_with_timeout,
    document_retry,
    embedding_retry,
    index_retry,
    managed_resource,
    network_retry,
    retry_with_context,
    safe_execute,
    safe_execute_async,
    with_fallback,
    with_retry,
)


# Create proper mock exception classes that support context
class DocumentLoadingError(Exception):
    def __init__(self, message="", context=None, **kwargs):
        super().__init__(message)
        self.context = context or {}


class EmbeddingError(Exception):
    def __init__(self, message="", context=None, **kwargs):
        super().__init__(message)
        self.context = context or {}


class IndexCreationError(Exception):
    def __init__(self, message="", context=None, **kwargs):
        super().__init__(message)
        self.context = context or {}


class ResourceError(Exception):
    def __init__(self, message="", context=None, **kwargs):
        super().__init__(message)
        self.context = context or {}


class RetryExhaustedError(Exception):
    def __init__(
        self, message="", operation="", context=None, original_error=None, **kwargs
    ):
        super().__init__(message)
        self.operation = operation
        self.context = context or {}
        self.original_error = original_error


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

        successful_function()

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

        eventually_successful_function()

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

        function_with_no_jitter()
        assert result == "success"
        assert call_count == 2

    @patch("utils.error_recovery.logger")
    def test_with_retry_context_preservation(self, mock_logger):
        """Test that retry preserves error context."""

        @with_retry(max_attempts=2, base_wait=0.01)
        def function_with_context_error():
            error = EmbeddingError("Test error", context={"model": "test"})
            raise error

        with pytest.raises(EmbeddingError):
            function_with_context_error()

    def test_with_retry_exponential_base(self):
        """Test retry with custom exponential base."""
        call_count = 0

        @with_retry(max_attempts=3, base_wait=0.01, exponential_base=3.0)
        def function_exponential_base():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Fail once")
            return "success"

        function_exponential_base()
        assert result == "success"

    def test_with_retry_reraise_disabled(self):
        """Test retry with reraise disabled."""

        @with_retry(max_attempts=2, base_wait=0.01, reraise=False)
        def always_failing():
            raise ValueError("Always fails")

        # Should not raise when reraise=False
        always_failing()
        assert result is None  # Default behavior when reraise is False


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

        primary_function()

        assert result == "primary"
        assert not fallback_called

    @patch("utils.error_recovery.logger")
    def test_with_fallback_primary_fails(self, mock_logger):
        """Test fallback when primary function fails."""

        def fallback_function():
            return "fallback_success"

        @with_fallback(fallback_function)
        def failing_primary():
            raise ConnectionError("Primary failed")

        failing_primary()

        assert result == "fallback_success"
        mock_logger.warning.assert_called_once()

    @patch("utils.error_recovery.logger")
    def test_with_fallback_both_fail(self, mock_logger):
        """Test when both primary and fallback functions fail."""

        def failing_fallback():
            raise RuntimeError("Fallback also failed")

        @with_fallback(failing_fallback)
        def failing_primary():
            raise ConnectionError("Primary failed")

        with pytest.raises(RuntimeError, match="Fallback also failed"):
            failing_primary()

        assert mock_logger.warning.called
        assert mock_logger.error.called

    @patch("utils.error_recovery.logger")
    def test_with_fallback_args_kwargs(self, mock_logger):
        """Test fallback with arguments and keyword arguments."""

        def fallback_function(*args, **kwargs):
            return f"fallback_args_{len(args)}_kwargs_{len(kwargs)}"

        @with_fallback(fallback_function)
        def failing_primary(*args, **kwargs):
            raise ValueError("Primary failed")

        failing_primary("arg1", "arg2", kwarg1="value1")

        assert result == "fallback_args_2_kwargs_1"


class TestSpecializedRetryDecorators:
    """Test suite for specialized retry decorators."""

    def test_embedding_retry_success(self):
        """Test embedding_retry decorator success."""

        @embedding_retry
        def embedding_operation():
            return "embedding_result"

        embedding_operation()
        assert result == "embedding_result"

    def test_embedding_retry_with_embedding_error(self):
        """Test embedding_retry with EmbeddingError."""
        call_count = 0

        @embedding_retry
        def failing_embedding():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise EmbeddingError("Temporary embedding error")
            return "success"

        failing_embedding()
        assert result == "success"
        assert call_count == 2

    def test_index_retry_with_index_error(self):
        """Test index_retry with IndexCreationError."""
        call_count = 0

        @index_retry
        def failing_index_creation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise IndexCreationError("Temporary index error")
            return "index_created"

        failing_index_creation()
        assert result == "index_created"
        assert call_count == 2

    def test_document_retry_with_document_error(self):
        """Test document_retry with DocumentLoadingError."""
        call_count = 0

        @document_retry
        def failing_document_load():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise DocumentLoadingError("Temporary document error")
            return "document_loaded"

        failing_document_load()
        assert result == "document_loaded"
        assert call_count == 2

    def test_network_retry_with_connection_error(self):
        """Test network_retry with ConnectionError."""
        call_count = 0

        @network_retry
        def failing_network_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network timeout")
            return "network_success"

        failing_network_call()
        assert result == "network_success"
        assert call_count == 3

    def test_network_retry_stop_conditions(self):
        """Test network_retry stops on specific exceptions."""

        @network_retry
        def function_with_stop_condition():
            raise KeyboardInterrupt("User interrupted")

        with pytest.raises(KeyboardInterrupt):
            function_with_stop_condition()


class TestAsyncWithTimeout:
    """Test suite for async_with_timeout decorator."""

    @pytest.mark.asyncio
    async def test_async_with_timeout_success(self):
        """Test async function completes within timeout."""

        @async_with_timeout(timeout_seconds=1.0)
        async def quick_async_function():
            await asyncio.sleep(0.1)
            return "completed"

        await quick_async_function()
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
    @patch("utils.error_recovery.logger")
    async def test_async_with_timeout_logging(self, mock_logger):
        """Test async timeout logging."""

        @async_with_timeout(timeout_seconds=0.1)
        async def timeout_function():
            await asyncio.sleep(0.5)

        with pytest.raises(TimeoutError):
            await timeout_function()

        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_with_timeout_custom_timeout(self):
        """Test async function with custom timeout."""

        @async_with_timeout(timeout_seconds=2.0)
        async def medium_async_function():
            await asyncio.sleep(0.5)
            return "completed"

        await medium_async_function()
        assert result == "completed"


class TestCircuitBreaker:
    """Test suite for CircuitBreaker pattern."""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state (normal operation)."""
        breaker = CircuitBreaker(failure_threshold=3)

        with breaker:
            pass

        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    def test_circuit_breaker_open_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=2)

        # First failure
        with pytest.raises(ValueError), breaker:
            raise ValueError("First failure")

        # Second failure - should open circuit
        with pytest.raises(ValueError), breaker:
            raise ValueError("Second failure")

        assert breaker.state == "OPEN"
        assert breaker.failure_count == 2

    def test_circuit_breaker_open_state_rejects_calls(self):
        """Test circuit breaker rejects calls when open."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)

        # Cause failure to open circuit
        with pytest.raises(ValueError), breaker:
            raise ValueError("Failure")

        # Next call should be rejected immediately
        with pytest.raises(ResourceError, match="Circuit breaker is OPEN"), breaker:
            pass

    @patch("time.time")
    def test_circuit_breaker_half_open_after_timeout(self, mock_time):
        """Test circuit breaker enters half-open state after timeout."""
        mock_time.side_effect = [0, 0, 0, 2.0]  # Simulate time progression

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)

        # Cause failure
        with pytest.raises(ValueError), breaker:
            raise ValueError("Failure")

        # Should enter half-open after timeout
        with breaker:
            pass

        assert breaker.state == "CLOSED"  # Should reset to closed on success

    @patch("time.time")
    def test_circuit_breaker_recovery_success(self, mock_time):
        """Test successful recovery resets circuit breaker."""
        mock_time.side_effect = [0, 0, 0, 2.0, 2.0]

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)

        # Cause failure
        with pytest.raises(RuntimeError), breaker:
            raise RuntimeError("Initial failure")

        # Successful recovery
        with breaker:
            pass  # Success

        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    def test_circuit_breaker_custom_exceptions(self):
        """Test circuit breaker with custom expected exceptions."""
        breaker = CircuitBreaker(
            failure_threshold=2, expected_exception=(ConnectionError, TimeoutError)
        )

        # ValueError should not count as failure
        with pytest.raises(ValueError), breaker:
            raise ValueError("Not counted")

        assert breaker.failure_count == 0
        assert breaker.state == "CLOSED"

        # ConnectionError should count
        with pytest.raises(ConnectionError), breaker:
            raise ConnectionError("Counted failure")

        assert breaker.failure_count == 1

    @patch("utils.error_recovery.logger")
    def test_circuit_breaker_logging(self, mock_logger):
        """Test circuit breaker logging behavior."""
        breaker = CircuitBreaker(failure_threshold=1)

        # Cause failure to open circuit
        with pytest.raises(RuntimeError), breaker:
            raise RuntimeError("Test failure")

        mock_logger.warning.assert_called_once()

        # Test state transitions are logged
        assert "Circuit breaker opened" in mock_logger.warning.call_args[0][0]


class TestManagedResource:
    """Test suite for managed_resource context manager."""

    def test_managed_resource_success(self):
        """Test managed resource with successful operation."""
        resource = MagicMock()
        factory = Mock(return_value=resource)

        with managed_resource(factory) as managed:
            assert managed is resource
            pass

        factory.assert_called_once()
        resource.close.assert_called_once()

    def test_managed_resource_with_exception(self):
        """Test managed resource cleanup with exception."""
        resource = MagicMock()
        factory = Mock(return_value=resource)

        with pytest.raises(ValueError), managed_resource(factory) as _:
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

    @patch("utils.error_recovery.logger")
    def test_managed_resource_cleanup_failure(self, mock_logger):
        """Test managed resource when cleanup fails."""
        resource = MagicMock()
        resource.close.side_effect = RuntimeError("Cleanup failed")
        factory = Mock(return_value=resource)

        # Should not raise exception, just log warning
        with managed_resource(factory):
            pass

        mock_logger.warning.assert_called_once()

    def test_managed_resource_factory_failure(self):
        """Test managed resource when factory fails."""
        factory = Mock(side_effect=RuntimeError("Factory failed"))

        with (
            pytest.raises(RuntimeError, match="Factory failed"),
            managed_resource(factory),
        ):
            pass

    def test_managed_resource_exit_method(self):
        """Test managed resource with object that has __exit__ method."""
        resource = MagicMock()
        resource.close = None  # Remove close method
        factory = Mock(return_value=resource)

        with managed_resource(factory):
            pass

        resource.__exit__.assert_called_once()


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

    @pytest.mark.asyncio
    async def test_async_managed_resource_timeout(self):
        """Test async managed resource cleanup timeout."""
        resource = AsyncMock()
        resource.aclose.side_effect = asyncio.sleep(1.0)  # Long cleanup
        factory = AsyncMock(return_value=resource)

        # Should timeout and continue
        async with async_managed_resource(factory, timeout_seconds=0.1):
            pass

    @pytest.mark.asyncio
    @patch("utils.error_recovery.logger")
    async def test_async_managed_resource_cleanup_failure(self, mock_logger):
        """Test async managed resource when cleanup fails."""
        resource = AsyncMock()
        resource.aclose.side_effect = RuntimeError("Async cleanup failed")
        factory = AsyncMock(return_value=resource)

        async with async_managed_resource(factory):
            pass

        mock_logger.warning.assert_called_once()


class TestRetryWithContext:
    """Test suite for retry_with_context decorator."""

    @patch("utils.error_recovery.logger")
    def test_retry_with_context_success(self, mock_logger):
        """Test retry_with_context with successful execution."""

        @retry_with_context("test_operation", max_attempts=3)
        def successful_operation():
            return "success"

        successful_operation()
        assert result == "success"

    @patch("utils.error_recovery.logger")
    def test_retry_with_context_with_retries(self, mock_logger):
        """Test retry_with_context with retries."""
        call_count = 0

        @retry_with_context("test_operation", max_attempts=3, context={"model": "test"})
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        eventually_successful()
        assert result == "success"
        assert call_count == 3

    @patch("utils.error_recovery.logger")
    def test_retry_with_context_exhausted(self, mock_logger):
        """Test retry_with_context when retries are exhausted."""

        @retry_with_context("failing_operation", max_attempts=2)
        def always_failing():
            raise RuntimeError("Always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            always_failing()

        error = exc_info.value
        assert error.operation == "failing_operation"
        assert error.context["max_attempts"] == 2
        assert isinstance(error.original_error, RuntimeError)

    @patch("utils.error_recovery.logger")
    def test_retry_with_context_custom_context(self, mock_logger):
        """Test retry_with_context with custom context."""
        context = {"database": "test_db", "timeout": 30}

        @retry_with_context("db_operation", context=context, max_attempts=2)
        def failing_db_op():
            raise ConnectionError("DB connection failed")

        with pytest.raises(RetryExhaustedError) as exc_info:
            failing_db_op()

        error = exc_info.value
        assert error.context["database"] == "test_db"
        assert error.context["timeout"] == 30


class TestSafeExecute:
    """Test suite for safe_execute function."""

    def test_safe_execute_success(self):
        """Test safe_execute with successful function."""

        def successful_function():
            return "success"

        safe_execute(successful_function)
        assert result == "success"

    @patch("utils.error_recovery.logger")
    def test_safe_execute_with_exception(self, mock_logger):
        """Test safe_execute with exception."""

        def failing_function():
            raise ValueError("Function failed")

        safe_execute(failing_function, default_value="default")

        assert result == "default"
        mock_logger.warning.assert_called_once()

    def test_safe_execute_no_logging(self):
        """Test safe_execute with logging disabled."""

        def failing_function():
            raise RuntimeError("Function failed")

        safe_execute(failing_function, log_errors=False, default_value="quiet_default")

        assert result == "quiet_default"

    @patch("utils.error_recovery.logger")
    def test_safe_execute_with_operation_name(self, mock_logger):
        """Test safe_execute with custom operation name."""

        def failing_function():
            raise OSError("System error")

        safe_execute(
            failing_function, operation_name="custom_operation", default_value="handled"
        )

        assert result == "handled"
        # Check that custom operation name is used in logging
        log_call = mock_logger.warning.call_args[0][0]
        assert "custom_operation" in log_call

    def test_safe_execute_lambda(self):
        """Test safe_execute with lambda function."""
        safe_execute(lambda: "lambda_result")
        assert result == "lambda_result"

        safe_execute(lambda: 1 / 0, default_value="division_error_handled")
        assert result == "division_error_handled"


class TestSafeExecuteAsync:
    """Test suite for safe_execute_async function."""

    @pytest.mark.asyncio
    async def test_safe_execute_async_success(self):
        """Test safe_execute_async with successful function."""

        async def successful_async():
            return "async_success"

        await safe_execute_async(successful_async)
        assert result == "async_success"

    @pytest.mark.asyncio
    @patch("utils.error_recovery.logger")
    async def test_safe_execute_async_with_exception(self, mock_logger):
        """Test safe_execute_async with exception."""

        async def failing_async():
            raise ConnectionError("Async function failed")

        await safe_execute_async(failing_async, default_value="async_default")

        assert result == "async_default"
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_safe_execute_async_timeout(self):
        """Test safe_execute_async with timeout."""

        async def slow_async():
            await asyncio.sleep(1.0)
            return "slow_result"

        await safe_execute_async(
            slow_async, timeout_seconds=0.1, default_value="timeout_handled"
        )

        assert result == "timeout_handled"

    @pytest.mark.asyncio
    async def test_safe_execute_async_sync_function(self):
        """Test safe_execute_async with sync function."""

        def sync_function():
            return "sync_in_async"

        await safe_execute_async(sync_function)
        assert result == "sync_in_async"

    @pytest.mark.asyncio
    @patch("utils.error_recovery.logger")
    async def test_safe_execute_async_custom_operation_name(self, mock_logger):
        """Test safe_execute_async with custom operation name."""

        async def failing_async():
            raise TimeoutError("Async timeout")

        await safe_execute_async(
            failing_async, operation_name="async_custom_op", default_value="handled"
        )

        assert result == "handled"
        log_call = mock_logger.warning.call_args[0][0]
        assert "async_custom_op" in log_call


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

        combined_function()
        assert result == "fallback_result"
        assert call_count == 2  # Retried once, then fallback

    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker combined with retry logic."""
        breaker = CircuitBreaker(failure_threshold=2)
        call_count = 0

        @with_retry(max_attempts=3, base_wait=0.01)
        def function_with_circuit_breaker():
            nonlocal call_count
            with breaker:
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("Fail until circuit opens")
                return "success"

        # Should eventually succeed before circuit opens
        function_with_circuit_breaker()
        assert result == "success"

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
                await async_retry_function()
                break
            except ConnectionError:
                if attempt == 2:  # Last attempt
                    raise
                call_count -= 1  # Reset for retry logic
                continue

        assert result == "async_success"


class TestErrorRecoveryEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_retry_with_zero_attempts(self):
        """Test retry behavior with zero attempts."""

        @with_retry(max_attempts=0)
        def zero_attempt_function():
            return "should_not_execute"

        # Should handle gracefully
        zero_attempt_function()

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

    def test_safe_execute_with_none_function(self):
        """Test safe_execute with None function."""
        safe_execute(None, default_value="none_handled")
        assert result == "none_handled"

    def test_managed_resource_none_factory(self):
        """Test managed resource with None factory."""
        with pytest.raises(TypeError), managed_resource(None):
            pass

    def test_fallback_with_none_fallback(self):
        """Test fallback decorator with None fallback function."""
        with pytest.raises(TypeError):

            @with_fallback(None)
            def test_function():
                pass
