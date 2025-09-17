"""Tests for OpenTelemetry setup helpers."""

from __future__ import annotations

from collections import defaultdict

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider

from src.config.settings import settings
from src.telemetry import opentelemetry as otel


def teardown_function() -> None:
    """Reset global providers after each test."""
    otel.shutdown_metrics()
    otel.shutdown_tracing()


def test_setup_tracing_disabled() -> None:
    """Tracing remains untouched when feature flag is disabled."""
    base_provider = trace.get_tracer_provider()
    disabled_settings = settings.model_copy(
        update={
            "observability": settings.observability.model_copy(
                update={"enabled": False}
            )
        }
    )
    otel.setup_tracing(disabled_settings)
    assert trace.get_tracer_provider() is base_provider
    assert otel._TRACE_PROVIDER is None


def test_setup_tracing_enabled() -> None:
    """Tracing provider is configured once when enabled."""
    enabled_settings = settings.model_copy(
        update={
            "observability": settings.observability.model_copy(
                update={
                    "enabled": True,
                    "protocol": "http/protobuf",
                    "endpoint": "http://localhost:4318/v1/traces",
                    "sampling_ratio": 0.5,
                }
            )
        }
    )
    otel.setup_tracing(enabled_settings)
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    assert otel._TRACE_PROVIDER is provider


def test_setup_metrics_enabled() -> None:
    """Metrics provider is configured with periodic exporter when enabled."""
    enabled_settings = settings.model_copy(
        update={
            "observability": settings.observability.model_copy(
                update={
                    "enabled": True,
                    "protocol": "http/protobuf",
                    "endpoint": "http://localhost:4318",
                    "metrics_interval_ms": 2000,
                }
            )
        }
    )
    otel.setup_metrics(enabled_settings)
    provider = metrics.get_meter_provider()
    assert isinstance(provider, MeterProvider)
    assert otel._METER_PROVIDER is provider


def test_configure_observability_instruments_llamaindex(monkeypatch) -> None:
    """LlamaIndex instrumentation registers once when enabled."""
    calls: defaultdict[str, int] = defaultdict(int)

    class DummyInstrumentor:
        def __init__(self) -> None:
            calls["init"] += 1

        def start_registering(self) -> None:
            calls["started"] += 1

        def shutdown(self) -> None:
            calls["shutdown"] += 1

    monkeypatch.setattr(otel, "LlamaIndexOpenTelemetry", DummyInstrumentor)
    enabled_settings = settings.model_copy(
        update={
            "observability": settings.observability.model_copy(
                update={
                    "enabled": True,
                    "endpoint": "http://localhost:4318",
                    "instrument_llamaindex": True,
                }
            )
        }
    )
    otel.configure_observability(enabled_settings)
    assert isinstance(otel._LLAMA_INDEX_INSTRUMENTOR, DummyInstrumentor)
    assert calls == {"init": 1, "started": 1}

    # Idempotent configuration should not instantiate additional instrumentors
    otel.configure_observability(enabled_settings)
    assert calls == {"init": 1, "started": 1}

    otel.shutdown_tracing()
    assert calls.get("shutdown") == 1
    assert otel._LLAMA_INDEX_INSTRUMENTOR is None
