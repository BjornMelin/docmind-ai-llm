"""Structured logging configuration using loguru for DocMind AI.

This module configures structured logging with loguru for comprehensive error
tracking, performance monitoring, and debugging capabilities. Provides:
- Console and file logging with rotation and retention
- JSON serialization for structured log parsing
- Contextual logging with request/operation IDs
- Performance metrics integration
- Security-conscious logging configuration

Example:
    Basic usage::

        from utils.logging_config import logger, setup_logging

        # Initialize logging
        setup_logging()

        # Use contextual logging
        with logger.contextualize(user_id="123", operation="index_creation"):
            logger.info("Starting document indexing")

        # Structured error logging
        try:
            risky_operation()
        except Exception as e:
            logger.error(
                "Operation failed",
                extra={
                    "error_type": type(e).__name__,
                    "context": {"operation": "indexing"},
                    "traceback": logger.opt(exception=True)
                }
            )

Attributes:
    logger: Configured loguru logger instance
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from models import AppSettings

settings = AppSettings()


def setup_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    json_logs: bool = True,
    log_directory: str = "logs",
) -> None:
    """Configure structured logging for the application.

    Sets up comprehensive logging with console and file handlers, JSON
    serialization for structured parsing, and security-conscious configuration
    that prevents credential exposure.

    Args:
        console_level: Minimum level for console output (DEBUG/INFO/WARNING/ERROR)
        file_level: Minimum level for file logging (DEBUG/INFO/WARNING/ERROR)
        json_logs: Whether to serialize logs as JSON for structured parsing
        log_directory: Directory for log file storage

    Features:
        - Automatic log rotation (10 MB files, 7 day retention)
        - JSON serialization for log aggregation systems
        - Contextual logging with operation IDs
        - Security filtering to prevent credential exposure
        - Performance metrics integration
        - Async-safe logging with enqueuing

    Example:
        >>> setup_logging(console_level="DEBUG", json_logs=True)
        >>> logger.info("Application started", extra={"version": "1.0.0"})
    """
    # Remove default handler to prevent duplicate logs
    logger.remove()

    # Determine log levels based on settings
    if hasattr(settings, "debug_mode") and settings.debug_mode:
        console_level = "DEBUG"
        file_level = "DEBUG"

    # Create log directory
    log_dir = Path(log_directory)
    log_dir.mkdir(exist_ok=True)

    # Security filter to prevent credential exposure
    def security_filter(record: dict[str, Any]) -> bool:
        """Filter out sensitive information from logs."""
        message = record.get("message", "").lower()

        # Block messages containing potential credentials
        sensitive_terms = [
            "password",
            "token",
            "secret",
            "key",
            "credential",
            "authorization",
            "bearer",
            "api_key",
            "private_key",
        ]

        if any(term in message for term in sensitive_terms):
            record["message"] = "[REDACTED - SENSITIVE INFORMATION]"

        # Filter sensitive data from extra fields
        extra = record.get("extra", {})
        if isinstance(extra, dict):
            for key in list(extra.keys()):
                if any(sensitive in key.lower() for sensitive in sensitive_terms):
                    extra[key] = "[REDACTED]"

        return True

    # Console handler with colors and structured format
    logger.add(
        sys.stderr,
        level=console_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=settings.debug_mode if hasattr(settings, "debug_mode") else False,
        enqueue=True,  # Thread-safe async logging
        filter=security_filter,
        catch=True,
    )

    # File handler with rotation and retention
    if json_logs:
        # JSON format for structured parsing
        logger.add(
            log_dir / "docmind_{time:YYYY-MM-DD}.json",
            level=file_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            serialize=True,  # JSON format
            enqueue=True,
            filter=security_filter,
            catch=True,
            format="{time} | {level} | {name}:{function}:{line} | {message}",
        )
    else:
        # Human-readable format
        logger.add(
            log_dir / "docmind_{time:YYYY-MM-DD}.log",
            level=file_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            enqueue=True,
            filter=security_filter,
            catch=True,
        )

    # Error-only file for critical issues
    logger.add(
        log_dir / "docmind_errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        rotation="5 MB",
        retention="14 days",  # Keep errors longer
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message} | "
            "{exception}"
        ),
        enqueue=True,
        filter=security_filter,
        backtrace=True,
        diagnose=True,  # Full diagnosis for errors
        catch=True,
    )

    # Add global context
    logger.configure(
        extra={
            "app": "docmind-ai",
            "version": "1.0.0",
            "environment": "development"
            if (hasattr(settings, "debug_mode") and settings.debug_mode)
            else "production",
        }
    )

    logger.info(
        "Structured logging initialized",
        extra={
            "console_level": console_level,
            "file_level": file_level,
            "json_logs": json_logs,
            "log_directory": str(log_dir.absolute()),
        },
    )


def get_logger(name: str | None = None) -> Any:
    """Get a named logger instance with structured context.

    Args:
        name: Logger name (defaults to caller's module name)

    Returns:
        Loguru logger bound with module context

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module started")
    """
    if name:
        return logger.bind(module=name)
    return logger


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics with structured data.

    Args:
        operation: Name of the operation being measured
        duration: Execution time in seconds
        **kwargs: Additional context data

    Example:
        >>> log_performance("index_creation", 45.2, doc_count=100, gpu=True)
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


def log_error_with_context(
    error: Exception, operation: str, context: dict[str, Any] | None = None, **kwargs
) -> None:
    """Log errors with comprehensive context information.

    Args:
        error: Exception instance
        operation: Name of the operation that failed
        context: Additional context dictionary
        **kwargs: Extra context fields

    Example:
        >>> try:
        ...     create_index()
        ... except Exception as e:
        ...     log_error_with_context(
        ...         e, "index_creation",
        ...         context={"doc_count": 100},
        ...         gpu_enabled=True
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


# Pre-configure logger for immediate use
if not logger._core.handlers:
    setup_logging()

# Export configured logger
__all__ = [
    "logger",
    "setup_logging",
    "get_logger",
    "log_performance",
    "log_error_with_context",
]
