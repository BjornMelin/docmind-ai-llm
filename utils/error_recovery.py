"""Error recovery and retry patterns using tenacity for DocMind AI.

This module provides robust retry mechanisms and error recovery strategies using
tenacity library for handling transient failures, network issues, and resource
constraints. Includes pre-configured retry patterns for common operations.

Features:
- Exponential backoff with jitter for network operations
- Circuit breaker patterns for external services
- Graceful fallback strategies
- Performance-aware retry limits
- Context-aware error handling
- Memory and resource-conscious retry logic

Example:
    Basic usage::

        from utils.error_recovery import (
            with_retry, with_fallback, embedding_retry,
            index_retry, async_with_timeout
        )

        # Simple retry with exponential backoff
        @with_retry(max_attempts=3)
        def create_embeddings(docs):
            return embedding_model.embed(docs)

        # Async operation with timeout and retries
        @async_with_timeout(timeout_seconds=30)
        @embedding_retry
        async def create_embeddings_async(docs):
            return await embedding_model.embed_async(docs)

        # Fallback strategy
        @with_fallback(lambda *args, **kwargs: default_embeddings())
        def create_embeddings_with_fallback(docs):
            return expensive_embedding_model.embed(docs)

Decorators:
    with_retry: General-purpose retry decorator
    with_fallback: Fallback strategy decorator
    embedding_retry: Optimized for embedding operations
    index_retry: Optimized for index creation
    document_retry: Optimized for document loading
    async_with_timeout: Async timeout wrapper
"""

import asyncio
import functools
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar

from tenacity import (
    RetryError,
    Retrying,
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

from utils.exceptions import (
    DocumentLoadingError,
    EmbeddingError,
    IndexCreationError,
    ResourceError,
    RetryExhaustedError,
)
from utils.logging_config import logger

T = TypeVar("T")


# Pre-configured retry strategies for common operations
def with_retry(
    max_attempts: int = 3,
    base_wait: float = 1.0,
    max_wait: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: tuple = (Exception,),
    stop_on: tuple = (),
    reraise: bool = True,
):
    """General-purpose retry decorator with exponential backoff and jitter.

    Args:
        max_attempts: Maximum number of retry attempts
        base_wait: Base wait time in seconds (minimum delay)
        max_wait: Maximum wait time in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to avoid thundering herd
        retry_on: Exception types to retry on
        stop_on: Exception types to immediately stop on
        reraise: Whether to reraise the original exception on final failure

    Returns:
        Decorated function with retry logic

    Example:
        >>> @with_retry(max_attempts=5, base_wait=2.0, max_wait=30.0)
        ... def flaky_network_call():
        ...     return requests.get("https://api.example.com/data")
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Build retry conditions
        retry_condition = retry_if_exception_type(retry_on)
        if stop_on:
            retry_condition = retry_condition & retry_if_not_exception_type(stop_on)

        # Choose wait strategy
        if jitter:
            wait_strategy = wait_random_exponential(
                multiplier=base_wait, max=max_wait, exp_base=exponential_base
            )
        else:
            wait_strategy = wait_exponential(
                multiplier=base_wait, max=max_wait, exp_base=exponential_base
            )

        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_strategy,
            retry=retry_condition,
            reraise=reraise,
            before_sleep=before_sleep_log(logger, "WARNING"),
            after=after_log(logger, "INFO"),
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add context to errors
                if hasattr(e, "context"):
                    e.context.update(
                        {
                            "retry_function": func.__name__,
                            "retry_max_attempts": max_attempts,
                            "retry_strategy": "exponential_backoff_with_jitter"
                            if jitter
                            else "exponential_backoff",
                        }
                    )
                raise

        return wrapper

    return decorator


def with_fallback(fallback_func: Callable[..., T]):
    """Decorator to add fallback behavior when primary function fails.

    Args:
        fallback_func: Function to call if primary function fails

    Returns:
        Decorated function with fallback logic

    Example:
        >>> @with_fallback(lambda docs: simple_embeddings(docs))
        ... def advanced_embeddings(docs):
        ...     return gpu_accelerated_embeddings(docs)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"{func.__name__} failed, using fallback: {fallback_func.__name__}",
                    extra={
                        "primary_function": func.__name__,
                        "fallback_function": fallback_func.__name__,
                        "error": str(e),
                    },
                )
                try:
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(
                        "Both primary and fallback functions failed",
                        extra={
                            "primary_error": str(e),
                            "fallback_error": str(fallback_error),
                        },
                    )
                    raise fallback_error

        return wrapper

    return decorator


# Specialized retry patterns for specific operations
embedding_retry = with_retry(
    max_attempts=3,
    base_wait=2.0,
    max_wait=30.0,
    retry_on=(EmbeddingError, ConnectionError, TimeoutError, ResourceError),
    stop_on=(KeyboardInterrupt, SystemExit, MemoryError),
    reraise=True,
)

index_retry = with_retry(
    max_attempts=5,
    base_wait=3.0,
    max_wait=60.0,
    retry_on=(IndexCreationError, ConnectionError, TimeoutError),
    stop_on=(KeyboardInterrupt, SystemExit),
    reraise=True,
)

document_retry = with_retry(
    max_attempts=3,
    base_wait=1.0,
    max_wait=15.0,
    retry_on=(DocumentLoadingError, FileNotFoundError, PermissionError),
    stop_on=(KeyboardInterrupt, SystemExit),
    reraise=True,
)

# Quick retry for transient network issues
network_retry = with_retry(
    max_attempts=5,
    base_wait=0.5,
    max_wait=10.0,
    retry_on=(ConnectionError, TimeoutError, OSError),
    stop_on=(KeyboardInterrupt, SystemExit),
    reraise=True,
)


def async_with_timeout(timeout_seconds: float = 30.0):
    """Add timeout to async functions with structured error handling.

    Args:
        timeout_seconds: Maximum execution time in seconds

    Returns:
        Decorated async function with timeout logic

    Example:
        >>> @async_with_timeout(timeout_seconds=60.0)
        ... async def long_running_operation():
        ...     await asyncio.sleep(45)
        ...     return "completed"
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), timeout=timeout_seconds
                )
            except TimeoutError:
                logger.error(
                    f"{func.__name__} timed out after {timeout_seconds}s",
                    extra={
                        "function": func.__name__,
                        "timeout_seconds": timeout_seconds,
                        "args_preview": str(args)[:100],
                    },
                )
                raise TimeoutError(
                    f"{func.__name__} exceeded {timeout_seconds}s timeout"
                ) from e

        return wrapper

    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation for external services.

    Prevents cascading failures by temporarily disabling calls to failing
    services and providing fast failures during outages.

    States:
        - CLOSED: Normal operation, calls are allowed
        - OPEN: Service is failing, calls are rejected immediately
        - HALF_OPEN: Testing if service has recovered

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        >>> with breaker:
        ...     result = external_api_call()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: tuple = (Exception,),
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Exception types that count as failures
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def __enter__(self):
        """Enter circuit breaker context."""
        current_time = time.time()

        if self.state == "OPEN":
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise ResourceError(
                    "Circuit breaker is OPEN",
                    context={
                        "failure_count": self.failure_count,
                        "time_until_retry": self.recovery_timeout
                        - (current_time - self.last_failure_time),
                        "state": self.state,
                    },
                )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit circuit breaker context and update state."""
        if exc_type is None:
            # Success - reset failure count
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED state")
        elif issubclass(exc_type, self.expected_exception):
            # Failure - increment count and potentially open circuit
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures",
                    extra={
                        "failure_threshold": self.failure_threshold,
                        "recovery_timeout": self.recovery_timeout,
                    },
                )


@contextmanager
def managed_resource(
    resource_factory: Callable,
    cleanup_func: Callable | None = None,
    timeout_seconds: float = 30.0,
):
    """Context manager for automatic resource cleanup with timeout.

    Args:
        resource_factory: Function that creates the resource
        cleanup_func: Optional cleanup function (defaults to .close() if available)
        timeout_seconds: Maximum time to wait for cleanup

    Example:
        >>> def create_client():
        ...     return SomeClient()
        >>>
        >>> with managed_resource(create_client) as client:
        ...     result = client.do_work()
    """
    resource = None
    try:
        resource = resource_factory()
        yield resource
    except Exception as e:
        logger.error(
            f"Error in managed resource: {e}",
            extra={"resource_type": type(resource).__name__ if resource else "unknown"},
        )
        raise
    finally:
        if resource is not None:
            try:
                # Try custom cleanup function first
                if cleanup_func:
                    cleanup_func(resource)
                # Fall back to common cleanup methods
                elif hasattr(resource, "close"):
                    resource.close()
                elif hasattr(resource, "__exit__"):
                    resource.__exit__(None, None, None)
            except Exception as cleanup_error:
                logger.warning(
                    f"Resource cleanup failed: {cleanup_error}",
                    extra={"resource_type": type(resource).__name__},
                )


@asynccontextmanager
async def async_managed_resource(
    resource_factory: Callable,
    cleanup_func: Callable | None = None,
    timeout_seconds: float = 30.0,
):
    """Async context manager for automatic resource cleanup.

    Args:
        resource_factory: Async function that creates the resource
        cleanup_func: Optional async cleanup function
        timeout_seconds: Maximum time to wait for cleanup

    Example:
        >>> async def create_client():
        ...     return await AsyncClient.create()
        >>>
        >>> async with async_managed_resource(create_client) as client:
        ...     result = await client.do_work()
    """
    resource = None
    try:
        if asyncio.iscoroutinefunction(resource_factory):
            resource = await resource_factory()
        else:
            resource = resource_factory()
        yield resource
    except Exception as e:
        logger.error(
            f"Error in async managed resource: {e}",
            extra={"resource_type": type(resource).__name__ if resource else "unknown"},
        )
        raise
    finally:
        if resource is not None:
            try:
                # Try custom cleanup function first
                if cleanup_func:
                    if asyncio.iscoroutinefunction(cleanup_func):
                        await asyncio.wait_for(
                            cleanup_func(resource), timeout=timeout_seconds
                        )
                    else:
                        cleanup_func(resource)
                # Fall back to common async cleanup methods
                elif hasattr(resource, "aclose"):
                    await asyncio.wait_for(resource.aclose(), timeout=timeout_seconds)
                elif hasattr(resource, "close") and asyncio.iscoroutinefunction(
                    resource.close
                ):
                    await asyncio.wait_for(resource.close(), timeout=timeout_seconds)
                elif hasattr(resource, "close"):
                    resource.close()
            except Exception as cleanup_error:
                logger.warning(
                    f"Async resource cleanup failed: {cleanup_error}",
                    extra={"resource_type": type(resource).__name__},
                )


def retry_with_context(
    operation_name: str,
    max_attempts: int = 3,
    context: dict[str, Any] | None = None,
    **retry_kwargs,
):
    """Retry decorator that includes operation context in errors.

    Args:
        operation_name: Name of the operation for logging and error context
        max_attempts: Maximum number of retry attempts
        context: Additional context to include in error messages
        **retry_kwargs: Additional arguments passed to tenacity retry

    Returns:
        Decorated function with contextual retry logic

    Example:
        >>> @retry_with_context("embedding_generation", max_attempts=3,
        ...                     context={"model": "bge-large"})
        ... def create_embeddings(docs):
        ...     return model.embed(docs)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_context = {
                "operation": operation_name,
                "function": func.__name__,
                "max_attempts": max_attempts,
                **(context or {}),
            }

            try:
                for attempt in Retrying(
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_random_exponential(multiplier=1, max=30),
                    before_sleep=before_sleep_log(logger, "WARNING"),
                    reraise=True,
                    **retry_kwargs,
                ):
                    with attempt:
                        logger.debug(
                            f"Attempting {operation_name} (attempt {attempt.retry_state.attempt_number})",
                            extra={"context": operation_context},
                        )
                        return func(*args, **kwargs)

            except RetryError as e:
                # Convert to our custom exception with full context
                raise RetryExhaustedError(
                    f"Operation '{operation_name}' failed after {max_attempts} attempts",
                    context=operation_context,
                    original_error=e.last_attempt.exception(),
                    operation=operation_name,
                ) from e

        return wrapper

    return decorator


# Convenience functions for common retry patterns
def safe_execute(
    func: Callable,
    default_value: Any = None,
    log_errors: bool = True,
    operation_name: str = None,
) -> Any:
    """Execute function safely with error handling and optional default value.

    Args:
        func: Function to execute
        default_value: Value to return if function fails
        log_errors: Whether to log errors
        operation_name: Name for logging context

    Returns:
        Function result or default value on failure

    Example:
        >>> result = safe_execute(
        ...     lambda: risky_operation(),
        ...     default_value=[],
        ...     operation_name="data_loading"
        ... )
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.warning(
                f"Safe execution failed: {operation_name or func.__name__}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "default_value": str(default_value)[:100],
                },
            )
        return default_value


async def safe_execute_async(
    func: Callable,
    default_value: Any = None,
    log_errors: bool = True,
    operation_name: str = None,
    timeout_seconds: float = 30.0,
) -> Any:
    """Execute async function safely with error handling and timeout.

    Args:
        func: Async function to execute
        default_value: Value to return if function fails
        log_errors: Whether to log errors
        operation_name: Name for logging context
        timeout_seconds: Maximum execution time

    Returns:
        Function result or default value on failure

    Example:
        >>> result = await safe_execute_async(
        ...     lambda: async_risky_operation(),
        ...     default_value=[],
        ...     operation_name="async_data_loading"
        ... )
    """
    try:
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(func(), timeout=timeout_seconds)
        else:
            return func()
    except Exception as e:
        if log_errors:
            logger.warning(
                f"Safe async execution failed: {operation_name or func.__name__}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "default_value": str(default_value)[:100],
                    "timeout_seconds": timeout_seconds,
                },
            )
        return default_value


# Export all retry utilities and decorators
__all__ = [
    "with_retry",
    "with_fallback",
    "embedding_retry",
    "index_retry",
    "document_retry",
    "network_retry",
    "async_with_timeout",
    "CircuitBreaker",
    "managed_resource",
    "async_managed_resource",
    "retry_with_context",
    "safe_execute",
    "safe_execute_async",
]
