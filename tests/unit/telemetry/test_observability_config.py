"""Unit tests for OpenTelemetry configuration helpers."""

from __future__ import annotations

from opentelemetry import metrics, trace

from src.config.settings import ObservabilityConfig, settings
from src.telemetry import opentelemetry as otel


def test_configure_observability_console(configured_observability_console) -> None:
    """Console exporters are registered when observability is enabled."""
    tracer_provider = trace.get_tracer_provider()
    meter_provider = metrics.get_meter_provider()
    # Console fixtures install SDK providers; verify they are not no-op stubs.
    assert tracer_provider.__class__.__name__ != "NoOpTracerProvider"
    assert meter_provider.__class__.__name__ != "NoOpMeterProvider"


def test_configure_observability_disabled(monkeypatch) -> None:
    """When disabled, configuration leaves tracer/meter providers untouched."""
    monkeypatch.setattr(otel, "_TRACE_PROVIDER", None)
    monkeypatch.setattr(otel, "_METER_PROVIDER", None)
    start_tracer = trace.get_tracer_provider()
    start_meter = metrics.get_meter_provider()

    original = settings.observability.model_copy()
    cfg = original.model_copy(update={"enabled": False})
    monkeypatch.setattr(settings, "observability", cfg, raising=False)
    otel.configure_observability(settings)

    assert trace.get_tracer_provider() is start_tracer
    assert metrics.get_meter_provider() is start_meter
    settings.observability = original


def test_observability_headers_formatted() -> None:
    """_build_otlp_kwargs formats headers into strings."""
    obs = ObservabilityConfig(
        enabled=True,
        endpoint="http://collector:4318",
        headers={"x-auth": "token"},
    )
    kwargs = otel._build_otlp_kwargs(obs)
    assert kwargs["headers"] == {"x-auth": "token"}
    assert kwargs["endpoint"] == "http://collector:4318"
