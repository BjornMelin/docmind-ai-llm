"""Telemetry utilities for DocMind."""

from .opentelemetry import (
    setup_metrics,
    setup_tracing,
    shutdown_metrics,
    shutdown_tracing,
)

__all__ = [
    "setup_metrics",
    "setup_tracing",
    "shutdown_metrics",
    "shutdown_tracing",
]
