---
spec: SPEC-012
title: Observability: OpenTelemetry Tracing & Metrics
version: 1.1.0
date: 2025-09-16
owners: ["ai-arch"]
status: Accepted
related_requirements:
  - NFR-OBS-001: Structured event logging
  - NFR-OBS-002: Trace & metric coverage for ingestion/retrieval/export
related_adrs: ["ADR-001","ADR-010","ADR-032"]
---

## Objective

Provide a unified OpenTelemetry configuration for DocMind AI covering traces, metrics, and structured events across ingestion, snapshot persistence, GraphRAG operations, and the Streamlit UI. Replace bespoke logging helpers with standards-based exporters and ensure deterministic instrumentation suitable for local-first deployments.

## Configuration

### ObservabilityConfig (src/config/settings.py)

```python
class ObservabilityConfig(BaseModel):
    enabled: bool = True
    service_name: str = "docmind-ai"
    endpoint: HttpUrl | None = None  # OTLP endpoint; None -> console exporters
    headers: dict[str, str] = Field(default_factory=dict)
    protocol: Literal["grpc", "http/protobuf"] = "http/protobuf"
    sampling_ratio: float = 1.0
    metrics_interval_ms: int = 30000
    export_timeout_ms: int = 10000
    resource_attributes: dict[str, str] = Field(default_factory=dict)
    enable_metrics: bool = True
    enable_traces: bool = True
```

- Values may be overridden by environment variables following the `DOCMIND_OBSERVABILITY__*` naming convention.
- Resource attributes SHALL merge user-supplied keys with defaults (`service.name`, `service.namespace`, `telemetry.sdk.*`).

### Dependencies

- The project exposes an optional extras group for observability (`uv sync --extra observability`) that installs OTLP HTTP/gRPC exporters and `portalocker` for cross-platform snapshot locking.
- Base environments retain console exporters so instrumentation remains optional during development.

### Environment Overrides

- `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_PROTOCOL`, and `OTEL_SERVICE_NAME` MUST be respected when present.
- When `ObservabilityConfig.enabled` is `False`, instrumentation MUST be disabled and no exporters registered.

## Implementation

### configure_observability(settings)

- Located in `src/telemetry/opentelemetry.py`.
- Responsibilities:
  - Build an OpenTelemetry `Resource` with merged attributes.
  - Instantiate a `TracerProvider` + `BatchSpanProcessor` (OTLP span exporter or console fallback).
  - Instantiate a `MeterProvider` + `PeriodicExportingMetricReader` (OTLP metric exporter or console fallback).
  - Register the providers globally via `opentelemetry.trace.set_tracer_provider` and `opentelemetry.metrics.set_meter_provider`.
  - Register `LlamaIndexOpenTelemetry` integration (if available) once per process.
  - Return a `ShutdownHandle` exposing `shutdown()` for tests.
- Idempotent: repeated calls reuse existing providers when configuration is unchanged and perform graceful shutdown when settings mutate.

### Instrumentation Points

| Component | Span / Metric | Attributes |
|-----------|---------------|------------|
| IngestionPipeline (`src/processing/ingestion_pipeline.py`) | `ingestion.pipeline.run` | `document_count`, `pipeline_id`, `cache_hit_ratio` |
| SnapshotManager (`src/persistence/snapshot.py`) | `snapshot.begin`, `snapshot.persist`, `snapshot.promote`, `snapshot.retention` | `snapshot_id`, `corpus_hash`, `config_hash`, `graph_export_count` |
| Graph export (`src/retrieval/graph_config.py`) | `graphrag.export` | `snapshot_id`, `export_format`, `seed_count`, `duration_ms` |
| RouterQueryEngine (`src/retrieval/router_factory.py`) | `router.select` | `route`, `pg_index_present`, `selector_type` |
| Streamlit UI actions (`src/pages/02_documents.py`, `src/pages/01_chat.py`) | `ui.action` | `action`, `success`, `snapshot_id` |

Metrics recorded via the shared `MeterProvider`:

- `snapshot.retention.count`
- `snapshot.promote.duration_ms`
- `ingestion.nodes.processed`
- `graph.export.duration_ms`
- `router.route.count`

Console exporters SHALL be used automatically when no OTLP endpoint is provided to preserve offline usability.

## Canonical Telemetry Events

Telemetry JSONL logs (written via `log_jsonl`) must include:

- `router_selected`: { route, selector_type, pg_index_present }
- `snapshot_stale_detected`: { reason, manifest_version, current_config_hash }
- `export_performed`: { kind, seed_count, capped, dest_relpath, duration_ms }
- `lock_takeover`: { lock_path, owner_id, elapsed_seconds, takeover_count }

## Acceptance Criteria

```gherkin
Feature: Observability bootstrap
  Scenario: Configure OTLP exporters
    Given ObservabilityConfig.enabled=true and endpoint=https://collector:4318
    When configure_observability(settings) runs
    Then traces and metrics SHALL be exported via OTLP HTTP/protobuf with configured headers
    And LlamaIndexOpenTelemetry SHALL be registered exactly once

  Scenario: Console fallback when endpoint missing
    Given ObservabilityConfig.enabled=true and endpoint=None
    When configure_observability(settings) runs
    Then console span and metric exporters SHALL be active
    And no network calls are attempted

Feature: Instrumented operations
  Scenario: Graph export emits telemetry
    Given a snapshot export succeeds
    Then `graphrag.export` span and `export_performed` event SHALL record seed_count and duration

Feature: Shutdown semantics
  Scenario: Observability disabled after enabling
    Given configure_observability(settings_enabled) has been called
    When configure_observability(settings_disabled) runs with enabled=false
    Then previously registered providers SHALL be shutdown without raising errors
```

## Testing Guidance

- Unit tests MUST patch OTLP exporters to avoid network access and assert exporter configuration.
- Integration tests SHOULD verify console output for offline mode and confirm spans exist using `InMemorySpanExporter`.
- ObservabilityConfig parsing is covered via `tests/unit/telemetry/test_observability_config.py`.
- UI and pipeline tests MUST assert that key actions push telemetry events using the shared helpers.
