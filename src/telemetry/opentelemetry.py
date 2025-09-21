"""OpenTelemetry scaffolding for DocMind agents.

Provides helpers to configure tracing and metrics exporters using application
settings. Exporters are only enabled when explicitly requested via
configuration to avoid implicit network dependencies during local
development/testing.
"""

from __future__ import annotations

import os
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, suppress
from importlib import metadata
from typing import Any, cast

try:
    from llama_index.observability.otel import LlamaIndexOpenTelemetry
except ImportError:  # pragma: no cover - optional dependency
    LlamaIndexOpenTelemetry = cast(Any, None)

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span

from src.config.settings import DocMindSettings, ObservabilityConfig

_OTEL_STATE: dict[str, Any] = {
    "trace_provider": None,
    "meter_provider": None,
    "instrumentor": None,
    "graph_metrics": {
        "counter": None,
        "duration": None,
        "seeds": None,
        "bytes": None,
    },
}


def _get_trace_provider() -> TracerProvider | None:
    provider = _OTEL_STATE.get("trace_provider")
    if isinstance(provider, TracerProvider):
        return provider
    return None


def _set_trace_provider(provider: TracerProvider | None) -> None:
    _OTEL_STATE["trace_provider"] = provider


def get_trace_provider() -> TracerProvider | None:
    """Return the active tracer provider if tracing is configured."""
    return _get_trace_provider()


def _get_meter_provider() -> MeterProvider | None:
    provider = _OTEL_STATE.get("meter_provider")
    if isinstance(provider, MeterProvider):
        return provider
    return None


def _set_meter_provider(provider: MeterProvider | None) -> None:
    _OTEL_STATE["meter_provider"] = provider


def get_meter_provider() -> MeterProvider | None:
    """Return the active meter provider if metrics are configured."""
    return _get_meter_provider()


def _get_instrumentor() -> Any:
    return _OTEL_STATE.get("instrumentor")


def _set_instrumentor(instrumentor: Any | None) -> None:
    _OTEL_STATE["instrumentor"] = instrumentor


def get_llamaindex_instrumentor() -> Any | None:
    """Return the configured LlamaIndex instrumentor if present."""
    return _get_instrumentor()


@contextmanager
def router_build_span(
    *,
    adapter_name: str,
    kg_requested: bool,
    hybrid_requested: bool,
) -> Generator[Span, None, None]:
    """Yield a span describing router construction."""
    tracer = trace.get_tracer(__name__)
    span = tracer.start_span("router_factory.build_router_engine")
    span.set_attribute("router.adapter_name", adapter_name)
    span.set_attribute("router.kg_requested", bool(kg_requested))
    span.set_attribute("router.hybrid_requested", bool(hybrid_requested))
    try:
        yield span
    finally:
        span.end()


def record_router_selection(
    span: Span,
    *,
    tool_names: Sequence[str],
    kg_enabled: bool,
) -> None:
    """Attach tool metadata to a router span."""
    span.set_attribute("router.kg_enabled", bool(kg_enabled))
    span.add_event(
        "router_selected",
        {
            "tool.count": len(tool_names),
            "tool.names": ",".join(sorted(tool_names)),
        },
    )


def _get_graph_metric(name: str) -> Any:
    graph_metrics = cast(dict[str, Any], _OTEL_STATE["graph_metrics"])
    return graph_metrics.get(name)


def _set_graph_metric(name: str, value: Any) -> None:
    graph_metrics = cast(dict[str, Any], _OTEL_STATE["graph_metrics"])
    graph_metrics[name] = value


def _reset_graph_metrics() -> None:
    graph_metrics = cast(dict[str, Any], _OTEL_STATE["graph_metrics"])
    for key in graph_metrics:
        graph_metrics[key] = None


def setup_tracing(app_settings: DocMindSettings) -> None:
    """Configure OpenTelemetry tracing when enabled.

    Args:
        app_settings: Loaded application settings.
    """
    obs = app_settings.observability
    if not obs.enabled:
        return

    if _get_trace_provider() is not None:
        return

    exporter = _create_span_exporter(obs)
    sampler = ParentBased(TraceIdRatioBased(obs.sampling_ratio))
    resource = _build_resource(app_settings)

    provider = TracerProvider(resource=resource, sampler=sampler)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _set_trace_provider(provider)


def setup_metrics(app_settings: DocMindSettings) -> None:
    """Configure OpenTelemetry metrics when enabled.

    Args:
        app_settings: Loaded application settings.
    """
    obs = app_settings.observability
    if not obs.enabled:
        return

    if _get_meter_provider() is not None:
        return

    exporter = _create_metric_exporter(obs)
    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=obs.metrics_interval_ms,
    )
    resource = _build_resource(app_settings)

    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    _set_meter_provider(provider)


def configure_observability(app_settings: DocMindSettings) -> None:
    """Idempotently configure OpenTelemetry tracing and metrics.

    Args:
        app_settings: Loaded application settings.
    """
    setup_tracing(app_settings)
    setup_metrics(app_settings)
    _maybe_configure_llamaindex(app_settings.observability)


def shutdown_tracing() -> None:
    """Shutdown the active tracer provider (used in tests)."""
    provider = _get_trace_provider()
    if provider is None:
        return
    provider.shutdown()
    _set_trace_provider(None)
    _shutdown_llamaindex()
    trace.set_tracer_provider(trace.NoOpTracerProvider())


def shutdown_metrics() -> None:
    """Shutdown the active meter provider (used in tests)."""
    provider = _get_meter_provider()
    if provider is None:
        return
    provider.shutdown()
    _set_meter_provider(None)
    metrics.set_meter_provider(metrics.NoOpMeterProvider())
    _reset_graph_metrics()


def record_graph_export_metric(
    export_type: str,
    *,
    duration_ms: float | None = None,
    seed_count: int | None = None,
    size_bytes: int | None = None,
    context: str | None = None,
) -> None:
    """Record OpenTelemetry metrics for graph exports when meters are configured."""
    if metrics is None:
        return
    meter = metrics.get_meter(__name__)
    counter = _get_graph_metric("counter")
    if counter is None:
        counter = meter.create_counter(
            "docmind.graph.export.count",
            description="Number of GraphRAG export operations",
        )
        _set_graph_metric("counter", counter)
    attributes: dict[str, str] = {"export_type": export_type}
    if context:
        attributes["context"] = context
    counter.add(1, attributes=attributes)
    if duration_ms is not None:
        duration_hist = _get_graph_metric("duration")
        if duration_hist is None:
            duration_hist = meter.create_histogram(
                "docmind.graph.export.duration",
                description="Graph export duration in milliseconds",
                unit="ms",
            )
            _set_graph_metric("duration", duration_hist)
        duration_hist.record(float(duration_ms), attributes=attributes)
    if seed_count is not None:
        seeds_hist = _get_graph_metric("seeds")
        if seeds_hist is None:
            seeds_hist = meter.create_histogram(
                "docmind.graph.export.seed_count",
                description="Number of seeds used for graph export",
                unit="1",
            )
            _set_graph_metric("seeds", seeds_hist)
        seeds_hist.record(float(seed_count), attributes=attributes)
    if size_bytes is not None:
        bytes_hist = _get_graph_metric("bytes")
        if bytes_hist is None:
            bytes_hist = meter.create_histogram(
                "docmind.graph.export.size",
                description="Graph export size in bytes",
                unit="By",
            )
            _set_graph_metric("bytes", bytes_hist)
        bytes_hist.record(float(size_bytes), attributes=attributes)


@contextmanager
def graph_export_span(
    *,
    adapter_name: str,
    fmt: str,
    depth: int,
    seed_count: int,
) -> Generator[Span, None, None]:
    """Yield a span to instrument GraphRAG export operations."""
    tracer = trace.get_tracer(__name__)
    span = tracer.start_span(f"graph_export.{fmt}")
    span.set_attribute("graph.export.adapter_name", adapter_name)
    span.set_attribute("graph.export.format", fmt)
    span.set_attribute("graph.export.depth", int(depth))
    span.set_attribute("graph.export.seed_count", int(seed_count))
    try:
        yield span
    finally:
        span.end()


def record_graph_export_event(
    span: Span,
    *,
    path: os.PathLike[str] | str,
    bytes_written: int,
) -> None:
    """Attach an export completion event to the span."""
    span.add_event(
        "export_performed",
        {
            "path": os.fspath(path),
            "size.bytes": int(bytes_written),
        },
    )


def _maybe_configure_llamaindex(obs: ObservabilityConfig) -> None:
    """Attach LlamaIndex OpenTelemetry instrumentation when configured."""
    if not obs.enabled or not obs.instrument_llamaindex:
        return
    if LlamaIndexOpenTelemetry is None:
        return
    if _get_instrumentor() is not None:
        return
    instrumentor = LlamaIndexOpenTelemetry()
    instrumentor.start_registering()
    _set_instrumentor(instrumentor)


def _shutdown_llamaindex() -> None:
    """Tear down LlamaIndex instrumentation if active."""
    instrumentor = _get_instrumentor()
    if instrumentor is None:
        return
    with suppress(Exception):
        shutdown = getattr(instrumentor, "shutdown", None)
        if callable(shutdown):
            shutdown()
    _set_instrumentor(None)


def _build_resource(app_settings: DocMindSettings) -> Resource:
    """Create OTEL resource metadata from application settings."""
    obs = app_settings.observability
    attributes: dict[str, Any] = {
        "service.name": obs.service_name,
    }
    with suppress(metadata.PackageNotFoundError):
        attributes["service.version"] = metadata.version("docmind_ai_llm")
    if environment := os.getenv("DOCMIND_ENVIRONMENT"):
        attributes["deployment.environment"] = environment
    return Resource.create(attributes)


def _create_span_exporter(obs: ObservabilityConfig) -> Any:
    """Instantiate an OTLP span exporter based on configured protocol."""
    kwargs = _build_otlp_kwargs(obs)
    if obs.protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GrpcSpanExporter,
        )

        return GrpcSpanExporter(**kwargs)

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpSpanExporter,
    )

    return HttpSpanExporter(**kwargs)


def _create_metric_exporter(obs: ObservabilityConfig) -> Any:
    """Instantiate an OTLP metric exporter based on configured protocol."""
    kwargs = _build_otlp_kwargs(obs)
    if obs.protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter as GrpcMetricExporter,
        )

        return GrpcMetricExporter(**kwargs)

    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter as HttpMetricExporter,
    )

    return HttpMetricExporter(**kwargs)


def _build_otlp_kwargs(obs: ObservabilityConfig) -> dict[str, Any]:
    """Return keyword arguments shared by metric and span exporters."""
    headers = _format_headers(obs.headers)
    kwargs: dict[str, Any] = {}
    if obs.endpoint:
        kwargs["endpoint"] = obs.endpoint
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
