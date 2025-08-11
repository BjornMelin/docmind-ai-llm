"""Minimal logging configuration for DocMind AI.

This module provides basic logging setup using loguru.
"""

from loguru import logger


def setup_logging(log_directory: str = "logs", debug_mode: bool = False) -> None:
    """Set up logging configuration."""
    pass


def get_logger():
    """Get the configured logger."""
    return logger


def log_error_with_context(message: str, context: dict = None) -> None:
    """Log error with context."""
    logger.error(message, extra=context or {})


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics."""
    logger.info(f"Performance: {operation} took {duration:.3f}s", extra=kwargs)


__all__ = [
    "logger",
    "setup_logging",
    "get_logger",
    "log_error_with_context",
    "log_performance",
]
