---
spec: SPEC-012
title: Observability: OpenTelemetry Tracing & Metrics
version: 1.2.0
date: 2026-01-09
owners: ["ai-arch"]
status: Accepted
related_requirements:
  - NFR-OBS-001: Structured event logging
  - NFR-OBS-002: Trace & metric coverage for ingestion/retrieval/export
related_adrs: ["ADR-001","ADR-010","ADR-032"]
---

## Objective

Provide local-first observability for DocMind AI:

- Structured **local JSONL telemetry** (sampling + rotation; never required network)
- Optional **OpenTelemetry tracing + metrics** export (explicitly enabled; safe-by-default)

## Configuration

### ObservabilityConfig (src/config/settings.py)

DocMind settings expose a single observability config surface under `settings.observability`.

| Field                   | Type                        |            Default | Notes                                                               |
| ----------------------- | --------------------------- | -----------------: | ------------------------------------------------------------------- |
| `enabled`               | `bool`                      |            `False` | Export is opt-in (offline-first).                                   |
| `service_name`          | `str`                       | `"docmind-agents"` | Used as OTEL `service.name`.                                        |
| `endpoint`              | `str \| None`               |             `None` | Optional exporter endpoint override (see endpoint semantics below). |
| `protocol`              | `"grpc" \| "http/protobuf"` |  `"http/protobuf"` | Which OTLP exporters to instantiate.                                |
| `headers`               | `dict[str,str]`             |               `{}` | Passed to exporters (auth, multi-tenant keys).                      |
| `sampling_ratio`        | `float`                     |              `1.0` | Trace sampling (ParentBased + TraceIdRatioBased).                   |
| `metrics_interval_ms`   | `int`                       |            `60000` | Passed to `PeriodicExportingMetricReader`.                          |
| `instrument_llamaindex` | `bool`                      |             `True` | Enables optional LlamaIndex OTel instrumentation when installed.    |

Environment overrides use Pydantic Settings V2 nested env keys:

- `DOCMIND_OBSERVABILITY__ENABLED=true`
- `DOCMIND_OBSERVABILITY__PROTOCOL=http/protobuf`
- `DOCMIND_OBSERVABILITY__ENDPOINT=http://localhost:4318/` (see endpoint semantics)
- `DOCMIND_OBSERVABILITY__HEADERS__x-auth=token`

Additional resource metadata:

- `DOCMIND_ENVIRONMENT` → sets OTEL resource attribute `deployment.environment` (when present).

### Dependencies

- OpenTelemetry SDK + OTLP exporters are installed in the base environment (`pyproject.toml` dependencies).
- LlamaIndex OTel instrumentation is optional and provided by the `observability` extra:
  - `uv sync --extra observability` (adds `llama-index-observability-otel`).

### Environment Overrides

DocMind passes `endpoint`/`headers` explicitly when configured, but the underlying OpenTelemetry exporters also respect standard OTEL environment variables when a value is not provided (for example `OTEL_EXPORTER_OTLP_ENDPOINT`, signal-specific endpoints, and OTEL metric interval/timeouts).

When `settings.observability.enabled` is `False`, DocMind does not install SDK providers and no exporters are registered.

### Endpoint Semantics (OTLP HTTP vs gRPC)

DocMind uses a _single_ `observability.endpoint` value for both traces and metrics exporters.

- For `protocol="grpc"` this is typically fine (one host:port for both signals).
- For `protocol="http/protobuf"`, traces and metrics normally use different paths (`/v1/traces` vs `/v1/metrics`). The OpenTelemetry Python HTTP exporters only auto-append these paths **when the endpoint argument is not provided**; however DocMind normalizes a configured `observability.endpoint` to the correct per-signal `/v1/<signal>` endpoint when the protocol is HTTP.

Practical outcomes:

- Setting `DOCMIND_OBSERVABILITY__ENDPOINT=http://localhost:4318` works for both traces and metrics.
- If `DOCMIND_OBSERVABILITY__ENDPOINT` already includes a `/v1/<signal>` suffix, DocMind rewrites it per exporter (`/v1/traces` for traces, `/v1/metrics` for metrics).
- You can still leave `DOCMIND_OBSERVABILITY__ENDPOINT` unset and rely on standard OTEL env vars (`OTEL_EXPORTER_OTLP_ENDPOINT`, signal-specific endpoints, etc.).

## Implementation

### OpenTelemetry bootstrap (`src/telemetry/opentelemetry.py`)

DocMind exposes a small bootstrap API:

- `configure_observability(settings)`:
  - Calls `setup_tracing(settings)` and `setup_metrics(settings)`.
  - Optionally registers LlamaIndex instrumentation once per process.
  - Safe to call repeatedly; providers are configured at most once (per process).
- `shutdown_tracing()` / `shutdown_metrics()`:
  - Used by tests to reset global providers and internal state.

Implementation notes:

- Tracing uses `TracerProvider(resource=..., sampler=ParentBased(TraceIdRatioBased(...)))` and a `BatchSpanProcessor` with an OTLP span exporter.
- Metrics uses `MeterProvider(resource=..., metric_readers=[PeriodicExportingMetricReader(...)])` with an OTLP metric exporter.

### Instrumentation Points

Spans are present in these code paths (no-op unless tracing is enabled):

| Component                                              | Span name                                                                                              | Attributes / events                                                                                                                                               |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ingestion (`src/processing/ingestion_pipeline.py`)     | `ingest_documents`                                                                                     | `docmind.document_count`                                                                                                                                          |
| Snapshot (`src/persistence/snapshot.py`)               | `snapshot.write_manifest` / `snapshot.finalize` / `snapshot.persist_vector` / `snapshot.persist_graph` | (no attributes currently)                                                                                                                                         |
| Router build (`src/retrieval/router_factory.py`)       | `router_factory.build_router_engine`                                                                   | attrs: `router.adapter_name`, `router.kg_requested`, `router.hybrid_requested`, `router.kg_enabled`; event: `router_selected` (`tool.count`, `tool.names`)        |
| Router tool (`src/agents/tools/router_tool.py`)        | `router_tool.query`                                                                                    | attrs: `router.query.length`, `router.engine.available`, `router.selected_strategy`, `router.success`, `router.latency_ms`, …                                     |
| Coordinator (`src/agents/coordinator.py`)              | `coordinator.process_query`                                                                            | attrs: `coordinator.thread_id`, `query.length`, `coordinator.workflow_timeout`, `coordinator.fallback`                                                            |
| Chat UI (`src/pages/01_chat.py`)                       | `chat.staleness_check`                                                                                 | attrs: `snapshot.id`, `snapshot.is_stale`                                                                                                                         |
| Graph export helper (`src/telemetry/opentelemetry.py`) | `graph_export.<fmt>`                                                                                   | attrs: `graph.export.adapter_name`, `graph.export.format`, `graph.export.depth`, `graph.export.seed_count`; event: `export_performed` (`file.name`, `size.bytes`) |

Metrics recorded via the shared `MeterProvider`:

- `docmind.graph.export.count` (counter; attrs include `export_type` and optional `context`)
- `docmind.graph.export.duration` (histogram, `ms`)
- `docmind.graph.export.seed_count` (histogram)
- `docmind.graph.export.size` (histogram, `By`)
- `docmind.coordinator.latency` (histogram, `s`; attr `success`)
- `docmind.coordinator.calls` (counter; attr `success`)

Metric recording is fail-open: `record_graph_export_metric(...)` is a no-op unless a meter provider has been configured.

## Canonical Telemetry Events

### Local JSONL (`src/utils/telemetry.py`)

DocMind emits local JSONL events to the path returned by
`get_telemetry_jsonl_path()` (default: `./logs/telemetry.jsonl`) via
`log_jsonl(...)`.

Controls (sourced from settings; can be set via env vars or `.env` per Pydantic Settings precedence; see SPEC-031):

- `DOCMIND_TELEMETRY_DISABLED` → disables event writes
- `DOCMIND_TELEMETRY_SAMPLE=0.0..1.0` → sampling rate
- `DOCMIND_TELEMETRY_ROTATE_BYTES=<int>` → size-based rotation (`.1` suffix)
- Advanced (derived from settings schema): `DOCMIND_TELEMETRY__JSONL_PATH=<path>`
  overrides the JSONL destination path.

Canonical event keys used by requirements/tests:

- Retrieval + fusion:
  - `retrieval.fusion_mode`, `retrieval.prefetch_dense_limit`, `retrieval.prefetch_sparse_limit`, `retrieval.fused_limit`, `retrieval.return_count`, `retrieval.latency_ms`
- Dedup:
  - `dedup.before`, `dedup.after`, `dedup.dropped`, `dedup.key`
- Rerank:
  - `rerank.stage`, `rerank.topk`, `rerank.latency_ms`, `rerank.timeout` (+ optional `rerank.batch_size`, `rerank.processed_count`, `rerank.processed_batches`)
- Routing event:
  - `router_selected: true`, plus `route`, `timing_ms`, and `traversal_depth` when `route=="knowledge_graph"`
- Snapshot staleness event:
  - `snapshot_stale_detected: true`, plus `snapshot_id`, `reason`
- Export event:
  - `export_performed: true`, plus `export_type`, `seed_count`, `dest_basename`, `context`, and `duration_ms` when available (optional: `size_bytes`, `capped`).

## Acceptance Criteria

```gherkin
Feature: Observability bootstrap
  Scenario: Disabled by default
    Given settings.observability.enabled=false
    When configure_observability(settings) runs
    Then global tracer and meter providers SHALL remain no-op

  Scenario: Enabled configures SDK providers once
    Given settings.observability.enabled=true
    When configure_observability(settings) runs
    Then SDK tracer and meter providers SHALL be installed
    And subsequent calls SHALL be idempotent

Feature: Instrumented operations
  Scenario: Router tool emits traversal depth for KG route
    Given router_tool selects knowledge_graph
    Then a JSONL event SHALL include traversal_depth (int)

Feature: Shutdown semantics
  Scenario: Tests can reset providers
    Given observability has been configured
    When shutdown_tracing and shutdown_metrics run
    Then providers SHALL reset without raising errors
```

## Testing Guidance

- Unit tests MUST avoid network access:
  - patch exporter constructors (see `tests/shared_fixtures.py::configured_observability_console`)
  - verify configuration and idempotency (`tests/unit/telemetry/test_otel_setup.py`)
- JSONL telemetry schema tests:
  - `tests/unit/telemetry/test_telemetry_schema_assertions.py`
  - `tests/unit/telemetry/test_rotation_sampling.py`
- Feature telemetry tests:
  - `tests/unit/agents/test_router_tool_telemetry.py` (traversal depth)
  - `tests/unit/ui/test_documents_snapshot_utils.py` (export event payloads)
