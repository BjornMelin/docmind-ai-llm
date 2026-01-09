# DocMind OTEL Metrics Quick Reference

This document lists the OpenTelemetry metrics emitted by DocMind and explains
how to inspect them locally without deploying Prometheus/Grafana.

## Metric Catalog

Metrics are only recorded when the OpenTelemetry SDK providers are configured
(see `docs/specs/spec-012-observability.md`).

| Metric | Type | Attributes | Description |
| ------ | ---- | ---------- | ----------- |
| `docmind.graph.export.count` | Counter | `export_type`, `context?` | Number of graph export operations. |
| `docmind.graph.export.duration` | Histogram (`ms`) | `export_type`, `context?` | Graph export duration. |
| `docmind.graph.export.seed_count` | Histogram | `export_type`, `context?` | Seeds used for graph export. |
| `docmind.graph.export.size` | Histogram (`By`) | `export_type`, `context?` | Size of exported artifact in bytes. |
| `docmind.coordinator.latency` | Histogram (`s`) | `success` | End-to-end coordinator latency. |
| `docmind.coordinator.calls` | Counter | `success` | Coordinator invocation count. |

## Lightweight Local Inspection

You can print recent metrics to stdout using the built-in console exporter demo
(no collector required):

```bash
uv run python scripts/demo_metrics_console.py
```

The script configures the OpenTelemetry SDK with a `ConsoleMetricExporter` and
records sample graph export metrics so you can verify metric wiring.

For live inspection while running DocMind, set the following environment
variables before launching the app:

```bash
export DOCMIND_OBSERVABILITY__ENABLED=true
export DOCMIND_OBSERVABILITY__PROTOCOL="http/protobuf"   # or "grpc"

# Recommended for OTLP/HTTP: set the base endpoint and let the OTEL exporters
# append /v1/traces and /v1/metrics (leave DOCMIND_OBSERVABILITY__ENDPOINT unset).
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/"
```

If you need LlamaIndex-specific instrumentation, install the optional extra:

```bash
uv sync --extra observability
```

For notebook experiments, you can import and call the helpers in
`src/telemetry/opentelemetry.py` (for example `record_graph_export_metric(...)`)
after configuring the OpenTelemetry SDK with a `ConsoleMetricExporter`.
