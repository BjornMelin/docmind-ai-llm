"""Quick OTEL metrics demo without external collectors."""

from __future__ import annotations

import time

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)

from src.telemetry.opentelemetry import record_graph_export_metric


def main() -> None:
    """Run a short-lived OTEL metrics loop for local inspection."""
    exporter = ConsoleMetricExporter()
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=1000)
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)

    record_graph_export_metric(
        "jsonl",
        duration_ms=135.0,
        seed_count=24,
        size_bytes=4096,
        context="demo",
    )
    record_graph_export_metric(
        "parquet",
        duration_ms=245.0,
        seed_count=16,
        size_bytes=8192,
        context="demo",
    )
    record_graph_export_metric(
        "jsonl",
        duration_ms=75.0,
        seed_count=8,
        size_bytes=1024,
        context="demo",
    )

    # Allow the periodic exporter to flush once
    time.sleep(1.2)
    reader.force_flush()
    provider.shutdown()


if __name__ == "__main__":
    main()
