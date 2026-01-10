"""Simple monitoring and logging utilities for DocMind AI.

This module provides essential performance monitoring and logging capabilities
with a focus on simplicity and core functionality. Removes complex patterns
and dataclasses in favor of simple functions and context managers.

Key features:
- Basic performance timing with context managers
- Simple structured logging functions
- Memory usage tracking
- Essential metrics collection
- Performance reporting
"""

import sys
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import psutil
from loguru import logger

from src.config import settings


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Setup basic logging configuration with Loguru.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path for file output
    """
    # Remove default handler
    logger.remove()

    # Add console handler with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )

    logger.info("Logging configured: level={}, file={}", log_level, log_file)


def log_error_with_context(
    error: Exception,
    operation: str,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Log errors with context information.

    Args:
        error: The exception that was raised
        operation: Name of the operation that failed
        context: Optional context dictionary
        **kwargs: Additional context as keyword arguments
    """
    error_context = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    if context:
        error_context.update(context)
    if kwargs:
        error_context.update(kwargs)

    # Emit context as part of message for test observability
    logger.error("Operation failed {}", error_context)


def log_performance(
    operation: str,
    duration_seconds: float,
    **metrics: Any,
) -> None:
    """Log performance metrics for an operation.

    Args:
        operation: Name of the operation
        duration_seconds: Duration in seconds
        **metrics: Additional metrics to log
    """
    perf_data = {
        "operation": operation,
        "duration_seconds": round(duration_seconds, 3),
        "duration_ms": round(duration_seconds * 1000, 1),
    }

    if metrics:
        perf_data.update(metrics)

    # Emit metrics dict in message for test observability
    logger.info("Performance metrics {}", perf_data)


@contextmanager
def performance_timer(
    operation: str, **context: Any
) -> Generator[dict[str, Any], None, None]:
    """Context manager for timing operations.

    Args:
        operation: Name of the operation being timed.
        **context: Additional context to log alongside metrics.

    Yields:
        Dict[str, Any]: Mutable metrics mapping updated during the operation.
    """
    start_time = time.perf_counter()
    process = None
    start_memory = 0.0
    try:
        process = psutil.Process()
        start_memory = (
            process.memory_info().rss / settings.monitoring.bytes_to_mb_divisor
        )
    except (OSError, psutil.Error):
        process = None
        start_memory = 0.0

    metrics: dict[str, Any] = {"operation": operation}
    metrics.update(context)

    success = False
    try:
        yield metrics
        success = True
    except Exception as exc:
        metrics["error"] = str(exc)
        raise
    finally:
        end_time = time.perf_counter()
        end_memory = start_memory
        if process is not None:
            try:
                end_memory = (
                    process.memory_info().rss / settings.monitoring.bytes_to_mb_divisor
                )
            except (OSError, psutil.Error):
                end_memory = start_memory

        duration = end_time - start_time
        memory_delta = end_memory - start_memory

        metrics["duration_seconds"] = round(duration, 3)
        metrics["memory_delta_mb"] = round(memory_delta, 2)
        metrics["success"] = success

        excluded = {"operation", "duration_seconds"}
        extra = {k: v for k, v in metrics.items() if k not in excluded}
        log_performance(operation=operation, duration_seconds=duration, **extra)


@asynccontextmanager
async def async_performance_timer(
    operation: str, **context: Any
) -> AsyncGenerator[dict[str, Any], None]:
    """Async context manager for timing operations.

    Args:
        operation: Name of the async operation being timed.
        **context: Additional context to log alongside metrics.

    Yields:
        Dict[str, Any]: Mutable metrics mapping updated during the operation.
    """
    start_time = time.perf_counter()
    process = None
    start_memory = 0.0
    try:
        process = psutil.Process()
        start_memory = (
            process.memory_info().rss / settings.monitoring.bytes_to_mb_divisor
        )
    except (OSError, psutil.Error):
        process = None
        start_memory = 0.0

    metrics: dict[str, Any] = {"operation": operation}
    metrics.update(context)

    success = False
    try:
        yield metrics
        success = True
    except Exception as exc:
        metrics["error"] = str(exc)
        raise
    finally:
        end_time = time.perf_counter()
        end_memory = start_memory
        if process is not None:
            try:
                end_memory = (
                    process.memory_info().rss / settings.monitoring.bytes_to_mb_divisor
                )
            except (OSError, psutil.Error):
                end_memory = start_memory

        duration = end_time - start_time
        memory_delta = end_memory - start_memory

        metrics["duration_seconds"] = round(duration, 3)
        metrics["memory_delta_mb"] = round(memory_delta, 2)
        metrics["success"] = success

        excluded = {"operation", "duration_seconds"}
        extra = {k: v for k, v in metrics.items() if k not in excluded}
        log_performance(operation=operation, duration_seconds=duration, **extra)


def get_memory_usage() -> dict[str, float]:
    """Get current memory usage information.

    Returns:
        Dictionary with memory usage metrics in MB
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": round(
                memory_info.rss / settings.monitoring.bytes_to_mb_divisor, 2
            ),
            "vms_mb": round(
                memory_info.vms / settings.monitoring.bytes_to_mb_divisor, 2
            ),
            "percent": round(process.memory_percent(), 2),
        }
    except (OSError, psutil.Error) as e:
        logger.warning("Failed to get memory usage: {}", e)
        return {"rss_mb": 0.0, "vms_mb": 0.0, "percent": 0.0}


def get_system_info() -> dict[str, Any]:
    """Get basic system information.

    Returns:
        Dictionary with system metrics
    """
    try:
        return {
            "cpu_percent": psutil.cpu_percent(
                interval=settings.monitoring.cpu_monitoring_interval
            ),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "load_average": psutil.getloadavg()
            if hasattr(psutil, "getloadavg")
            else None,
        }
    except (OSError, psutil.Error) as e:
        logger.warning("Failed to get system info: {}", e)
        return {}


class SimplePerformanceMonitor:
    """Simple performance monitor for tracking operations."""

    def __init__(self) -> None:
        """Initialize the performance monitor with empty metrics list."""
        self.metrics: list[dict[str, Any]] = []

    def record_operation(
        self,
        operation: str,
        duration_seconds: float,
        success: bool = True,
        **metrics: Any,
    ) -> None:
        """Record metrics for a completed operation.

        Args:
            operation: Name of the operation
            duration_seconds: Duration in seconds
            success: Whether operation succeeded
            **metrics: Additional metrics
        """
        record = {
            "operation": operation,
            "duration_seconds": round(duration_seconds, 3),
            "success": success,
            "timestamp": time.time(),
        }
        record.update(metrics)

        self.metrics.append(record)
        log_performance(**record)

    def get_summary(self, operation: str | None = None) -> dict[str, Any]:
        """Get performance summary.

        Args:
            operation: Optional operation name to filter by

        Returns:
            Dictionary with performance summary
        """
        metrics = self.metrics
        if operation:
            metrics = [m for m in metrics if m["operation"] == operation]

        if not metrics:
            return {"total_operations": 0}

        durations = [m["duration_seconds"] for m in metrics]
        successes = [m for m in metrics if m.get("success", True)]

        return {
            "total_operations": len(metrics),
            "successful_operations": len(successes),
            "success_rate": len(successes)
            / len(metrics)
            * settings.monitoring.percent_multiplier,
            "avg_duration_seconds": sum(durations) / len(durations),
            "min_duration_seconds": min(durations),
            "max_duration_seconds": max(durations),
            "total_duration_seconds": sum(durations),
        }

    def clear_metrics(self) -> None:
        """Clear stored metrics."""
        self.metrics.clear()
        logger.info("Performance metrics cleared")


# Global performance monitor instance
_performance_monitor = SimplePerformanceMonitor()


def get_performance_monitor() -> SimplePerformanceMonitor:
    """Get the global performance monitor instance.

    Returns:
        SimplePerformanceMonitor instance
    """
    return _performance_monitor
