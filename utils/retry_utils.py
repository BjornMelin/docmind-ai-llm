"""Modern retry utilities using tenacity for DocMind AI.

This module provides clean, library-first retry patterns using tenacity decorators
with domain-specific exception handling. Replaces custom retry logic with proven
tenacity patterns for better maintainability and performance.

Features:
- Standard tenacity retry decorators
- Pre-configured retry patterns for common operations
- Domain-specific exception handling (DocMindError hierarchy)
- Simple fallback patterns
- Async timeout support
- Circuit breaker using tenacity stopping conditions

Example:
    Basic usage::

        from utils.retry_utils import (
            standard_retry, llm_retry, embedding_retry,
            with_fallback, async_with_timeout
        )

        # Simple retry with exponential backoff
        @embedding_retry
        def create_embeddings(docs):
            return embedding_model.embed(docs)

        # Async operation with timeout
        @async_with_timeout(timeout_seconds=30)
        @llm_retry
        async def llm_call_async(prompt):
            return await llm_client.generate(prompt)

        # Fallback strategy
        @with_fallback(lambda: default_result())
        def risky_operation():
            return expensive_operation()

Decorators:
    standard_retry: General-purpose retry with exponential backoff
    llm_retry: Optimized for LLM API calls with rate limiting
    embedding_retry: Optimized for embedding generation
    index_retry: Optimized for index creation operations
    document_retry: Optimized for document loading
    network_retry: Fast retry for network operations
"""

import asyncio
import functools
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar

from loguru import logger
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.exceptions import (
    AgentError,
    DocumentLoadingError,
    EmbeddingError,
    IndexCreationError,
    ResourceError,
)

T = TypeVar("T")

# Standard retry patterns using tenacity
standard_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(logger, "WARNING"),
    after=after_log(logger, "INFO"),
)

# LLM-specific retry with longer backoff for rate limits
llm_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, AgentError)),
    before_sleep=before_sleep_log(logger, "WARNING"),
)

# Embedding retry pattern
embedding_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(
        (EmbeddingError, ConnectionError, TimeoutError, ResourceError)
    ),
    before_sleep=before_sleep_log(logger, "WARNING"),
)

# Index creation retry pattern
index_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=3, min=3, max=60),
    retry=retry_if_exception_type((IndexCreationError, ConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(logger, "WARNING"),
)

# Document loading retry pattern
document_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=retry_if_exception_type(
        (DocumentLoadingError, FileNotFoundError, PermissionError)
    ),
    before_sleep=before_sleep_log(logger, "WARNING"),
)

# Fast network retry for transient issues
network_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(logger, "WARNING"),
)


def with_fallback(fallback_func):
    """Simple fallback decorator when primary function fails.

    Args:
        fallback_func: Function to call if primary function fails

    Returns:
        Decorated function with fallback logic

    Example:
        @with_fallback(lambda: "default")
        def risky_operation():
            return expensive_call()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"{func.__name__} failed, using fallback",
                    extra={"error": str(e), "fallback": fallback_func.__name__},
                )
                return fallback_func(*args, **kwargs)

        return wrapper

    return decorator


def async_with_timeout(timeout_seconds: float = 30.0):
    """Add timeout to async functions.

    Args:
        timeout_seconds: Maximum execution time in seconds

    Returns:
        Decorated async function with timeout logic

    Example:
        @async_with_timeout(timeout_seconds=60.0)
        async def slow_operation():
            await asyncio.sleep(45)
            return "completed"
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), timeout=timeout_seconds
                )
            except TimeoutError as e:
                logger.error(
                    f"{func.__name__} timed out after {timeout_seconds}s",
                    extra={"function": func.__name__, "timeout": timeout_seconds},
                )
                raise TimeoutError(
                    f"{func.__name__} exceeded {timeout_seconds}s timeout"
                ) from e

        return wrapper

    return decorator


def safe_execute(
    func: Callable,
    default_value: Any = None,
    log_errors: bool = True,
    operation_name: str | None = None,
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
        result = safe_execute(
            lambda: risky_operation(),
            default_value=[],
            operation_name="data_loading"
        )
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
    operation_name: str | None = None,
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


# Simple managed resource context manager using standard patterns


@contextmanager
def managed_resource(
    resource_factory: Callable,
    cleanup_func: Callable | None = None,
):
    """Simple context manager for resource cleanup.

    Args:
        resource_factory: Function that creates the resource
        cleanup_func: Optional cleanup function

    Example:
        with managed_resource(create_client) as client:
            result = client.do_work()
    """
    resource = None
    try:
        resource = resource_factory()
        yield resource
    finally:
        if resource is not None:
            try:
                if cleanup_func:
                    cleanup_func(resource)
                elif hasattr(resource, "close"):
                    resource.close()
            except Exception as e:
                logger.warning(f"Resource cleanup failed: {e}")


@asynccontextmanager
async def async_managed_resource(
    resource_factory: Callable,
    cleanup_func: Callable | None = None,
):
    """Simple async context manager for resource cleanup.

    Args:
        resource_factory: Function that creates the resource
        cleanup_func: Optional cleanup function

    Example:
        async with async_managed_resource(create_client) as client:
            result = await client.do_work()
    """
    resource = None
    try:
        if asyncio.iscoroutinefunction(resource_factory):
            resource = await resource_factory()
        else:
            resource = resource_factory()
        yield resource
    finally:
        if resource is not None:
            try:
                if cleanup_func:
                    if asyncio.iscoroutinefunction(cleanup_func):
                        await cleanup_func(resource)
                    else:
                        cleanup_func(resource)
                elif hasattr(resource, "aclose"):
                    await resource.aclose()
                elif hasattr(resource, "close"):
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
            except Exception as e:
                logger.warning(f"Async resource cleanup failed: {e}")


# Rate-limited retry patterns combining tenacity with rate limiting


# Rate limiting integration removed - use standard retry patterns for local app


# Export all retry utilities
__all__ = [
    "standard_retry",
    "llm_retry",
    "embedding_retry",
    "index_retry",
    "document_retry",
    "network_retry",
    "with_fallback",
    "async_with_timeout",
    "safe_execute",
    "safe_execute_async",
    "managed_resource",
    "async_managed_resource",
]
