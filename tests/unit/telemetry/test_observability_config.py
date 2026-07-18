"""Unit tests for OpenTelemetry configuration helpers."""

from __future__ import annotations

from opentelemetry import metrics, trace

from src.config.settings import ObservabilityConfig, settings
from src.telemetry import opentelemetry as otel


def test_configure_observability_console(configured_observability_console) -> None:
    """Console exporters are registered when observability is enabled."""
    tracer_provider = otel.get_trace_provider()
    meter_provider = otel.get_meter_provider()
    # Console fixtures install SDK providers; verify they are not no-op stubs.
    assert tracer_provider is not None
    assert meter_provider is not None
    assert tracer_provider.__class__.__name__ != "NoOpTracerProvider"
    assert meter_provider.__class__.__name__ != "NoOpMeterProvider"


def test_configure_observability_disabled(monkeypatch) -> None:
    """When disabled, configuration leaves tracer/meter providers untouched."""
    start_tracer = trace.NoOpTracerProvider()
    start_meter = metrics.NoOpMeterProvider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: start_tracer)
    monkeypatch.setattr(metrics, "get_meter_provider", lambda: start_meter)

    def _unexpected_provider(_provider: object) -> None:
        raise AssertionError("disabled configuration replaced a provider")

    monkeypatch.setattr(trace, "set_tracer_provider", _unexpected_provider)
    monkeypatch.setattr(metrics, "set_meter_provider", _unexpected_provider)

    original = settings.observability.model_copy()
    cfg = original.model_copy(update={"enabled": False})
    monkeypatch.setattr(settings, "observability", cfg, raising=False)
    otel.configure_observability(settings)

    assert trace.get_tracer_provider() is start_tracer
    assert metrics.get_meter_provider() is start_meter


def test_observability_headers_formatted() -> None:
    """_build_otlp_kwargs formats headers into strings."""
    obs = ObservabilityConfig(
        enabled=True,
        endpoint="http://collector:4318",
        headers={"x-auth": "token"},
    )
    kwargs = otel._build_otlp_kwargs(obs, signal="traces")
    assert kwargs["headers"] == {"x-auth": "token"}
    assert kwargs["endpoint"] == "http://collector:4318/v1/traces"

    kwargs_metrics = otel._build_otlp_kwargs(obs, signal="metrics")
    assert kwargs_metrics["endpoint"] == "http://collector:4318/v1/metrics"
