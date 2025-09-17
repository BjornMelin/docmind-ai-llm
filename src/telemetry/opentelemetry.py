"""OpenTelemetry scaffolding for DocMind agents.

Provides helpers to configure tracing and metrics exporters using application
settings. Exporters are only enabled when explicitly requested via
configuration to avoid implicit network dependencies during local
development/testing.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import suppress
from importlib import metadata
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from src.config.settings import DocMindSettings

_TRACE_PROVIDER: TracerProvider | None = None
_METER_PROVIDER: MeterProvider | None = None

_GRAPH_EXPORT_COUNTER = None
_GRAPH_EXPORT_DURATION = None
_GRAPH_EXPORT_SEEDS = None
_GRAPH_EXPORT_BYTES = None


def setup_tracing(app_settings: DocMindSettings) -> None:
    """Configure OpenTelemetry tracing when enabled.

    Args:
        app_settings: Loaded application settings.
    """
    if not app_settings.otel_enabled:
        return

    global _TRACE_PROVIDER  # pylint: disable=global-statement
    if _TRACE_PROVIDER is not None:
        return

    exporter = _create_span_exporter(app_settings)
    sampler = ParentBased(TraceIdRatioBased(app_settings.otel_sampling_ratio))
    resource = _build_resource(app_settings)

    provider = TracerProvider(resource=resource, sampler=sampler)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACE_PROVIDER = provider


def setup_metrics(app_settings: DocMindSettings) -> None:
    """Configure OpenTelemetry metrics when enabled.

    Args:
        app_settings: Loaded application settings.
    """
    if not app_settings.otel_enabled:
        return

    global _METER_PROVIDER  # pylint: disable=global-statement
    if _METER_PROVIDER is not None:
        return

    exporter = _create_metric_exporter(app_settings)
    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=app_settings.otel_metrics_interval_ms,
    )
    resource = _build_resource(app_settings)

    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    _METER_PROVIDER = provider


def configure_observability(app_settings: DocMindSettings) -> None:
    """Idempotently configure OpenTelemetry tracing and metrics.

    Args:
        app_settings: Loaded application settings.
    """
    setup_tracing(app_settings)
    setup_metrics(app_settings)


def shutdown_tracing() -> None:
    """Shutdown the active tracer provider (used in tests)."""
    global _TRACE_PROVIDER  # pylint: disable=global-statement
    if _TRACE_PROVIDER is None:
        return
    _TRACE_PROVIDER.shutdown()
    _TRACE_PROVIDER = None
    trace.set_tracer_provider(trace.NoOpTracerProvider())


def shutdown_metrics() -> None:
    """Shutdown the active meter provider (used in tests)."""
    global _METER_PROVIDER  # pylint: disable=global-statement
    if _METER_PROVIDER is None:
        return
    _METER_PROVIDER.shutdown()
    _METER_PROVIDER = None
    metrics.set_meter_provider(metrics.NoOpMeterProvider())


def record_graph_export_metric(
    export_type: str,
    *,
    duration_ms: float | None = None,
    seed_count: int | None = None,
    size_bytes: int | None = None,
    context: str | None = None,
) -> None:
    """Record OpenTelemetry metrics for graph exports when meters are configured."""
    if metrics is None:  # type: ignore[truthy-function]
        return
    meter = metrics.get_meter(__name__)
    global _GRAPH_EXPORT_COUNTER  # pylint: disable=global-statement
    global _GRAPH_EXPORT_DURATION
    global _GRAPH_EXPORT_SEEDS
    global _GRAPH_EXPORT_BYTES
    if _GRAPH_EXPORT_COUNTER is None:
        _GRAPH_EXPORT_COUNTER = meter.create_counter(
            "docmind.graph.export.count",
            description="Number of GraphRAG export operations",
        )
    attributes: dict[str, str] = {"export_type": export_type}
    if context:
        attributes["context"] = context
    _GRAPH_EXPORT_COUNTER.add(1, attributes=attributes)
    if duration_ms is not None:
        if _GRAPH_EXPORT_DURATION is None:
            _GRAPH_EXPORT_DURATION = meter.create_histogram(
                "docmind.graph.export.duration",
                description="Graph export duration in milliseconds",
                unit="ms",
            )
        _GRAPH_EXPORT_DURATION.record(float(duration_ms), attributes=attributes)
    if seed_count is not None:
        if _GRAPH_EXPORT_SEEDS is None:
            _GRAPH_EXPORT_SEEDS = meter.create_histogram(
                "docmind.graph.export.seed_count",
                description="Number of seeds used for graph export",
                unit="1",
            )
        _GRAPH_EXPORT_SEEDS.record(float(seed_count), attributes=attributes)
    if size_bytes is not None:
        if _GRAPH_EXPORT_BYTES is None:
            _GRAPH_EXPORT_BYTES = meter.create_histogram(
                "docmind.graph.export.size",
                description="Graph export size in bytes",
                unit="By",
            )
        _GRAPH_EXPORT_BYTES.record(float(size_bytes), attributes=attributes)


def _build_resource(app_settings: DocMindSettings) -> Resource:
    """Create OTEL resource metadata from application settings."""
    attributes: dict[str, Any] = {
        "service.name": app_settings.otel_service_name,
    }
    with suppress(metadata.PackageNotFoundError):
        attributes["service.version"] = metadata.version("docmind_ai_llm")
    if environment := os.getenv("DOCMIND_ENVIRONMENT"):
        attributes["deployment.environment"] = environment
    return Resource.create(attributes)


def _create_span_exporter(app_settings: DocMindSettings):
    """Instantiate an OTLP span exporter based on configured protocol."""
    kwargs = _build_otlp_kwargs(app_settings)
    if app_settings.otel_exporter_protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GrpcSpanExporter,
        )

        return GrpcSpanExporter(**kwargs)

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpSpanExporter,
    )

    return HttpSpanExporter(**kwargs)


def _create_metric_exporter(app_settings: DocMindSettings):
    """Instantiate an OTLP metric exporter based on configured protocol."""
    if app_settings.otel_exporter_endpoint == "console":
        return ConsoleMetricExporter()
    kwargs = _build_otlp_kwargs(app_settings)
    if app_settings.otel_exporter_protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter as GrpcMetricExporter,
        )

        return GrpcMetricExporter(**kwargs)

    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter as HttpMetricExporter,
    )

    return HttpMetricExporter(**kwargs)


def _build_otlp_kwargs(app_settings: DocMindSettings) -> dict[str, Any]:
    """Return keyword arguments shared by metric and span exporters."""
    headers = _format_headers(app_settings.otel_exporter_headers)
    kwargs: dict[str, Any] = {}
    if app_settings.otel_exporter_endpoint:
        kwargs["endpoint"] = app_settings.otel_exporter_endpoint
    if headers:
        kwargs["headers"] = headers
    return kwargs


def _format_headers(headers: Mapping[str, Any]) -> dict[str, str]:
    """Convert mapping values to strings as expected by OTLP exporters."""
    return {str(key): str(value) for key, value in headers.items()}


__all__ = [
    "configure_observability",
    "record_graph_export_metric",
    "setup_metrics",
    "setup_tracing",
    "shutdown_metrics",
    "shutdown_tracing",
]
