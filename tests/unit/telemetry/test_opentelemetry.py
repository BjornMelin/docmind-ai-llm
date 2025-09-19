"""Tests for the OpenTelemetry scaffolding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.telemetry import opentelemetry as otel


@pytest.fixture(autouse=True)
def reset_globals() -> None:
    otel.shutdown_tracing()
    otel.shutdown_metrics()
    otel._LLAMA_INDEX_INSTRUMENTOR = None  # type: ignore[attr-defined]
    otel._TRACE_PROVIDER = None  # type: ignore[attr-defined]
    otel._METER_PROVIDER = None  # type: ignore[attr-defined]
    otel._GRAPH_EXPORT_COUNTER = None  # type: ignore[attr-defined]
    otel._GRAPH_EXPORT_DURATION = None  # type: ignore[attr-defined]
    otel._GRAPH_EXPORT_SEEDS = None  # type: ignore[attr-defined]
    otel._GRAPH_EXPORT_BYTES = None  # type: ignore[attr-defined]


@pytest.mark.unit
def test_setup_tracing_configures_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class DummyProvider:
        def __init__(self, resource: object, sampler: object) -> None:
            captured["resource"] = resource
            captured["sampler"] = sampler
            captured["provider"] = self

        def add_span_processor(self, processor: object) -> None:
            captured["processor"] = processor

        def shutdown(self) -> None:  # pragma: no cover - cleanup path
            captured["shutdown"] = True

    class DummyProcessor:
        def __init__(self, exporter: object) -> None:
            captured["exporter"] = exporter
            self.exporter = exporter

    stored_provider: list[object] = []

    def _set_tracer_provider(provider: object) -> None:
        stored_provider.append(provider)

    monkeypatch.setattr(otel, "TracerProvider", DummyProvider)
    monkeypatch.setattr(otel, "BatchSpanProcessor", DummyProcessor)
    monkeypatch.setattr(
        otel, "trace", SimpleNamespace(set_tracer_provider=_set_tracer_provider)
    )
    monkeypatch.setattr(otel, "ParentBased", lambda sampler: ("parent", sampler))
    monkeypatch.setattr(otel, "TraceIdRatioBased", lambda ratio: ("ratio", ratio))
    monkeypatch.setattr(otel, "_create_span_exporter", lambda obs: "exporter")
    monkeypatch.setattr(otel, "_build_resource", lambda settings: "resource")

    settings = SimpleNamespace(
        observability=SimpleNamespace(
            enabled=True, sampling_ratio=0.5, instrument_llamaindex=False
        ),
        app_version="test",
    )

    otel.setup_tracing(settings)

    assert stored_provider
    assert stored_provider[0] is captured["provider"]
    assert captured["processor"].exporter == "exporter"
    assert captured["resource"] == "resource"
    assert captured["sampler"] == ("parent", ("ratio", 0.5))


@pytest.mark.unit
def test_record_graph_export_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    records: dict[str, list] = {"counter": [], "duration": [], "seeds": [], "bytes": []}

    class DummyCounter:
        def add(self, value: int, attributes: dict[str, str] | None = None) -> None:
            records["counter"].append((value, attributes))

    class DummyHistogram:
        def __init__(self, bucket: str) -> None:
            self.bucket = bucket

        def record(
            self, value: float, attributes: dict[str, str] | None = None
        ) -> None:
            records[self.bucket].append((value, attributes))

    class DummyMeter:
        def create_counter(self, *_args, **_kwargs) -> DummyCounter:
            return DummyCounter()

        def create_histogram(self, name: str, **_kwargs) -> DummyHistogram:
            bucket = {
                "docmind.graph.export.duration": "duration",
                "docmind.graph.export.seed_count": "seeds",
                "docmind.graph.export.size": "bytes",
            }[name]
            return DummyHistogram(bucket)

    class DummyMetricsModule:
        def __init__(self) -> None:
            self.meter = DummyMeter()

        def get_meter(self, *_args, **_kwargs) -> DummyMeter:
            return self.meter

    monkeypatch.setattr(otel, "metrics", DummyMetricsModule())

    otel.record_graph_export_metric(
        "graph",
        duration_ms=12.5,
        seed_count=4,
        size_bytes=1024,
        context="unit",
    )

    assert records["counter"] == [(1, {"export_type": "graph", "context": "unit"})]
    assert records["duration"] == [(12.5, {"export_type": "graph", "context": "unit"})]
    assert records["seeds"] == [(4.0, {"export_type": "graph", "context": "unit"})]
    assert records["bytes"] == [(1024.0, {"export_type": "graph", "context": "unit"})]
