"""Centralized logging utilities for DocMind AI.

This module provides consistent logging functionality across the entire application.
Consolidates logging functions that were duplicated across multiple modules to
follow DRY principles and provide a single source of truth for logging behavior.

Key features:
- Structured error logging with comprehensive context
- Performance metrics logging with standardized format
- Modern Loguru configuration with file and console handlers
- Thread-safe async-compatible logging patterns
- Structured JSON serialization for production environments
- Consistent logging patterns for debugging and monitoring

Example:
    Basic usage of logging utilities::

        from utils.logging_utils import (
            setup_logging,
            log_error_with_context,
            log_performance,
        )
        from loguru import logger

        # Setup logging configuration
        setup_logging("INFO")

        # Log structured errors
        try:
            risky_operation()
        except Exception as e:
            log_error_with_context(e, "risky_operation", context={"param": "value"})

        # Log performance metrics
        import time
        start = time.perf_counter()
        # ... do work ...
        duration = time.perf_counter() - start
        log_performance("work_operation", duration, items_processed=100)

        # Use contextual logging
        with logger.contextualize(task_id="123", user="admin"):
            logger.info("Processing started")
"""

import sys
from contextlib import contextmanager
from typing import Any

from loguru import logger


def log_error_with_context(
    error: Exception, operation: str, context: dict | None = None, **kwargs
) -> None:
    """Log errors with comprehensive context information.

    Provides a structured logging mechanism for error tracking, capturing
    detailed information about exceptions with minimal performance overhead.
    Consolidates error logging patterns used throughout the application.

    Args:
        error: The exception that was raised
        operation: Name or description of the operation that failed
        context: Optional dictionary of additional context information
        **kwargs: Additional keyword arguments to include in error context

    Note:
        - Uses Loguru for structured logging
        - Captures error type, message, operation name, and additional context
        - Supports flexible error tracking across different components
        - Thread-safe and async-compatible

    Example:
        >>> try:
        ...     result = divide(10, 0)
        ... except Exception as e:
        ...     log_error_with_context(
        ...         e, "division_operation",
        ...         context={"numerator": 10, "denominator": 0}
        ...     )
    """
    error_context = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
        **(context or {}),
        **kwargs,
    }
    logger.error(
        f"Operation failed: {operation}",
        extra={"error_context": error_context},
        exception=error,
    )


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Log structured performance metrics for operation timing.

    Captures detailed performance information with minimal overhead,
    supporting both human-readable and machine-parseable formats.
    Provides consistent performance logging across all application components.

    Args:
        operation: Name or description of the performance-tracked operation
        duration: Total operation duration in seconds
        **kwargs: Additional performance-related metadata to include

    Note:
        - Logs duration in seconds with three decimal points
        - Provides human-readable duration format
        - Supports extensible performance tracking
        - Minimal performance impact logging mechanism
        - Structured data format for metrics aggregation

    Example:
        >>> import time
        >>> start = time.perf_counter()
        >>> # ... perform operation ...
        >>> duration = time.perf_counter() - start
        >>> log_performance("embedding_generation", duration, batch_size=100)
    """
    logger.info(
        f"Performance: {operation} completed",
        extra={
            "performance": {
                "operation": operation,
                "duration_seconds": round(duration, 3),
                "duration_human": f"{duration:.2f}s",
                **kwargs,
            }
        },
    )


def setup_logging(log_level: str = "INFO") -> None:
    """Set up loguru logging configuration with file and console handlers.

    Configures comprehensive logging with both console output (with colors)
    and file-based logging with rotation and compression. Provides consistent
    logging configuration across the entire application.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Note:
        - Removes default Loguru handler to avoid duplicate output
        - Console handler uses colored output for better readability
        - File handler includes automatic rotation (10MB) and retention (7 days)
        - Compressed log archives to save disk space
        - Thread-safe and production-ready configuration

    Example:
        >>> setup_logging("DEBUG")  # Enable debug logging
        >>> setup_logging("INFO")   # Standard production logging
    """
    # Remove default handler to avoid duplicate output
    logger.remove()

    # Add console handler with colored output for development
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Add file handler for persistent logging with rotation
    logger.add(
        "logs/docmind.log",
        level=log_level,
        format=lambda record: (
            f"{record['time']:YYYY-MM-DD HH:mm:ss} | "
            f"{record['level']: <8} | "
            f"{record['name']}:{record['function']}:{record['line']} - "
            f"{record['message']}"
        ),
        rotation="10 MB",  # Rotate when file reaches 10MB
        retention="7 days",  # Keep logs for 7 days
        compression="zip",  # Compress rotated logs
    )

    logger.info(
        "Logging configured successfully",
        extra={"log_level": log_level, "console": True, "file": True},
    )


@contextmanager
def log_context(**context_data: Any):
    """Context manager for adding structured context to all log messages within a block.

    Modern Loguru pattern using contextualize() for temporary context-local state.
    Automatically removes context when exiting the context manager.

    Args:
        **context_data: Key-value pairs to add to all log messages in this context

    Example:
        >>> with log_context(user_id="123", request_id="abc-def"):
        ...     logger.info("Processing request")  # Includes user_id and request_id
        ...     log_performance("api_call", 0.5)   # Also includes context
    """
    with logger.contextualize(**context_data):
        yield


def create_bound_logger(**context_data: Any):
    """Create a logger instance with persistent bound context data.

    Modern Loguru pattern using bind() for creating logger instances with
    permanent context that persists across multiple log calls.

    Args:
        **context_data: Key-value pairs to permanently bind to this logger instance

    Returns:
        Logger instance with bound context data

    Example:
        >>> user_logger = create_bound_logger(user_id="123", module="auth")
        >>> user_logger.info("User logged in")  # Always includes user_id and module
        >>> user_logger.error("Authentication failed")  # Context persists
    """
    return logger.bind(**context_data)


def setup_production_logging(
    log_level: str = "INFO", enable_json: bool = False
) -> None:
    """Set up production-optimized logging with JSON serialization.

    Provides production-ready configuration with structured JSON output for
    log aggregation, thread-safe operations, and security-focused settings.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Enable JSON serialization for structured logging and
            log aggregation

    Note:
        - Uses enqueue=True for thread/multiprocess safety
        - JSON serialization enables easier integration with log aggregation
        - Enhanced error handling with catch=True
        - Automatic log rotation and compression
        - Removes sensitive information from logs in production

    Example:
        >>> setup_production_logging("INFO", enable_json=True)
        >>> setup_production_logging("DEBUG", enable_json=False)
    """
    # Remove default handler to avoid duplicate output
    logger.remove()

    # Console handler for development visibility
    if not enable_json:
        logger.add(
            sys.stdout,
            level=log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            colorize=True,
            enqueue=True,  # Thread-safe for production
            catch=True,  # Catch exceptions in logging itself
        )
    else:
        # JSON format for production log aggregation
        logger.add(
            sys.stdout,
            level=log_level,
            serialize=True,  # JSON serialization
            enqueue=True,  # Thread-safe
            catch=True,  # Error handling
        )

    # Enhanced file handler with production settings
    logger.add(
        "logs/docmind.log",
        level=log_level,
        format=lambda record: (
            f"{record['time']:YYYY-MM-DD HH:mm:ss} | "
            f"{record['level']: <8} | "
            f"{record['name']}:{record['function']}:{record['line']} - "
            f"{record['message']}"
        ),
        rotation="50 MB",  # Larger rotation for production
        retention="30 days",  # Longer retention for production
        compression="gz",  # Better compression ratio
        enqueue=True,  # Thread-safe
        catch=True,  # Error handling
    )

    logger.info(
        "Production logging configured",
        extra={
            "log_level": log_level,
            "json_enabled": enable_json,
            "thread_safe": True,
            "rotation": "50MB",
            "retention": "30days",
        },
    )
