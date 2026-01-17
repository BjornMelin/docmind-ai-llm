"""Monitoring helpers bridging JSONL telemetry and optional OpenTelemetry.

This module provides essential performance monitoring and logging capabilities
with a focus on simplicity and core functionality. Removes complex patterns
and dataclasses in favor of simple functions and context managers.

Key features:
- Basic performance timing with context managers
- Structured JSONL telemetry emission
- Optional OTEL metric recording (when enabled)
- Memory usage tracking
- Essential metrics collection
- Performance reporting
"""

import sys
import time
import warnings
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import psutil
from loguru import logger

from src.config import settings
from src.utils.log_safety import build_pii_log_entry
from src.utils.telemetry import log_jsonl

_OTEL_INSTRUMENTS: dict[str, Any] = {
    "meter": None,
    "duration_ms": None,
    "count": None,
}
_WARNED_DEPRECATED_RECORD_OPERATION = False
_WARNED_DEPRECATED_GET_MONITOR = False
warnings.filterwarnings("once", category=DeprecationWarning, module=__name__)


def _make_otel_log_patcher(
    enabled: bool,
) -> Callable[[Any], None]:
    """Create a Loguru patcher that injects OTEL trace/span IDs.

    Args:
        enabled: When False, inject empty fields but do not attempt OTEL lookups.

    Returns:
        A Loguru patcher function.
    """

    def _patch(record: Any) -> None:
        if not isinstance(record, dict):
            return
        extra = record.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            record["extra"] = extra
        extra.setdefault("otelTraceID", "")
        extra.setdefault("otelSpanID", "")
        extra.setdefault("otelTraceSampled", "false")
        if not enabled:
            return
        try:
            from opentelemetry import trace as _trace

            ctx = _trace.get_current_span().get_span_context()
            if ctx is None or not getattr(ctx, "is_valid", False):
                return
            extra["otelTraceID"] = f"{int(ctx.trace_id):032x}"
            extra["otelSpanID"] = f"{int(ctx.span_id):016x}"
            extra["otelTraceSampled"] = (
                "true" if bool(getattr(ctx.trace_flags, "sampled", False)) else "false"
            )
        except Exception:
            return

    return _patch


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure Loguru sinks and OTEL trace correlation fields.

    Args:
        log_level: Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Optional log file path for file output.
    """
    # Remove default handler
    logger.remove()

    obs_cfg = getattr(settings, "observability", None)
    obs_enabled = (
        bool(getattr(obs_cfg, "enabled", False))
        and float(getattr(obs_cfg, "sampling_ratio", 1.0) or 0.0) > 0.0
    )
    logger.configure(patcher=_make_otel_log_patcher(obs_enabled))

    prefix = ""
    if obs_enabled:
        prefix = (
            "trace_id={extra[otelTraceID]} span_id={extra[otelSpanID]} "
            "sampled={extra[otelTraceSampled]} | "
        )

    # Add console handler with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            f"{prefix}"
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
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                f"{prefix}"
                "{name}:{function}:{line} - {message}"
            ),
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
    redaction = build_pii_log_entry(str(error), key_id=operation or "exception")
    error_context = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_redacted": redaction.redacted,
        "error_fingerprint": redaction.fingerprint,
    }

    if context:
        error_context.update(context)
    if kwargs:
        error_context.update(kwargs)

    safe_keys = {"operation", "error_type", "error_redacted", "error_fingerprint"}
    op_key = operation or "exception"
    safe_context = {}
    for key, value in error_context.items():
        if key in safe_keys or not isinstance(value, str):
            safe_context[key] = value
            continue
        safe_context[key] = build_pii_log_entry(
            str(value), key_id=f"{op_key}:{key}"
        ).redacted
    log_jsonl({"error_logged": True, **safe_context})
    logger.error("Operation failed {}", safe_context)


def _record_otel_performance(operation: str, duration_ms: float) -> None:
    """Record basic OTEL metrics for an operation when metrics are configured.

    Args:
        operation: Operation name.
        duration_ms: Duration in milliseconds.
    """
    try:
        from src.telemetry.opentelemetry import get_meter_provider

        if get_meter_provider() is None:
            return
        from opentelemetry import metrics
    except Exception:
        return

    if _OTEL_INSTRUMENTS["meter"] is None:
        _OTEL_INSTRUMENTS["meter"] = metrics.get_meter(__name__)
    meter = _OTEL_INSTRUMENTS["meter"]
    if _OTEL_INSTRUMENTS["count"] is None:
        _OTEL_INSTRUMENTS["count"] = meter.create_counter(
            "docmind.operation.count",
            description="Number of recorded operations",
        )
    if _OTEL_INSTRUMENTS["duration_ms"] is None:
        _OTEL_INSTRUMENTS["duration_ms"] = meter.create_histogram(
            "docmind.operation.duration",
            description="Recorded operation duration in milliseconds",
            unit="ms",
        )
    attrs = {"operation": operation}
    _OTEL_INSTRUMENTS["count"].add(1, attributes=attrs)
    _OTEL_INSTRUMENTS["duration_ms"].record(float(duration_ms), attributes=attrs)


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

    log_jsonl({"performance_logged": True, **perf_data})
    _record_otel_performance(
        operation=operation, duration_ms=float(perf_data["duration_ms"])
    )
    logger.debug("Performance metrics {}", perf_data)


@contextmanager
def performance_timer(operation: str, **context: Any) -> Generator[dict[str, Any]]:
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
        redaction = build_pii_log_entry(str(exc), key_id=operation or "exception")
        metrics["error_redacted"] = redaction.redacted
        metrics["error_fingerprint"] = redaction.fingerprint
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
) -> AsyncGenerator[dict[str, Any]]:
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
        redaction = build_pii_log_entry(str(exc), key_id=operation or "exception")
        metrics["error_redacted"] = redaction.redacted
        metrics["error_fingerprint"] = redaction.fingerprint
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
    except (OSError, psutil.Error) as exc:
        redaction = build_pii_log_entry(str(exc), key_id="monitoring.get_memory_usage")
        logger.warning(
            "Failed to get memory usage (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
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
    except (OSError, psutil.Error) as exc:
        redaction = build_pii_log_entry(str(exc), key_id="monitoring.get_system_info")
        logger.warning(
            "Failed to get system info (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        return {}


class SimplePerformanceMonitor:
    """Deprecated: in-memory performance monitor for tracking operations.

    Prefer OTEL metrics (src.telemetry.opentelemetry) and JSONL telemetry events.
    """

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

        global _WARNED_DEPRECATED_RECORD_OPERATION
        if not _WARNED_DEPRECATED_RECORD_OPERATION:
            warnings.warn(
                (
                    "SimplePerformanceMonitor is deprecated; prefer OTEL metrics and "
                    "JSONL telemetry."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            _WARNED_DEPRECATED_RECORD_OPERATION = True
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
    global _WARNED_DEPRECATED_GET_MONITOR
    if not _WARNED_DEPRECATED_GET_MONITOR:
        warnings.warn(
            (
                "get_performance_monitor is deprecated; prefer OTEL metrics and JSONL "
                "telemetry."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        _WARNED_DEPRECATED_GET_MONITOR = True
    return _performance_monitor
