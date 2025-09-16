# DocMind OTEL Metrics Quick Reference

This document lists the OpenTelemetry metrics emitted by the ingestion refactor
and explains how to inspect them locally without deploying Prometheus/Grafana.

## Metric Catalog

| Metric | Type | Labels | Description |
| ------ | ---- | ------ | ----------- |
| `document_ingestion_duration_ms` | Histogram | `stage`, `status`, `strategy`, `cache_backend` | Stage-level ingestion latency in milliseconds. |
| `document_ingestion_errors_total` | Counter | `stage`, `error_type`, `cache_backend` | Count of failed ingestion attempts per stage/error. |
| `cache_operations_total` | Counter | `backend`, `op`, `outcome`, `strategy?` | Cache lifecycle events (writes, purges, TTL). |
| `ocr_pages_processed_total` | Counter | `mode`, `reason` | Pages processed per OCR policy decision. |

Additional spans are emitted for `ingestion.process` (attributes: `ingestion.strategy`, `ingestion.ocr_reason`).

## Lightweight Local Inspection

You can print recent metrics to stdout using the built-in console exporter demo:

```bash
uv run python scripts/demo_metrics_console.py
```

The script configures the OpenTelemetry SDK with a console exporter and records
sample ingestion/cache/OCR events so you can verify exporter wiring.

For live inspection while running DocMind, set the following environment
variables before launching the app:

```bash
export DOCMIND_OTEL_ENABLED=true
export DOCMIND_OTEL_EXPORTER_PROTOCOL="http/protobuf"
export DOCMIND_OTEL__EXPORTER_ENDPOINT="console"  # special value handled by DocMind
```

When `otel_enabled` is true and the endpoint is `console`, metrics are exported
via the console exporter at the interval defined by
`DOCMIND_OTEL__METRICS_INTERVAL_MS` (defaults to 60000 ms). This keeps
observability local-first and avoids introducing Prometheus/Grafana
dependencies.

For notebook experiments, you can copy the instrumentation helpers from
`src/telemetry/instrumentation.py` and call `record_ingestion_duration`,
`record_cache_operation`, or `record_ocr_decision` directly inside a notebook
cell once you configure the SDK with a `ConsoleMetricExporter`.
