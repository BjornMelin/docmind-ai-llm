"""Quick OTEL metrics demo without external collectors."""

from __future__ import annotations

import time

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)

from src.telemetry.instrumentation import (
    record_cache_operation,
    record_ingestion_duration,
    record_ocr_decision,
)


def main() -> None:
    """Run a short-lived OTEL metrics loop for local inspection."""
    exporter = ConsoleMetricExporter()
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=1000)
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)

    record_ingestion_duration(
        245.0, attributes={"strategy": "hi_res", "cache_backend": "sqlite"}
    )
    record_ingestion_duration(
        530.0,
        status="error",
        attributes={
            "strategy": "ocr_only",
            "cache_backend": "sqlite",
            "error_type": "ProcessingError",
        },
    )
    record_cache_operation(
        backend="sqlite", op="write", outcome="ok", attributes={"strategy": "hi_res"}
    )
    record_cache_operation(
        backend="sqlite", op="purge", outcome="ok", attributes={"reason": "ttl"}
    )
    record_ocr_decision(
        mode="hi_res", pages=7, attributes={"reason": "PDF_LARGE_SCANNED"}
    )

    # Allow the periodic exporter to flush once
    time.sleep(1.2)
    reader.force_flush()
    provider.shutdown()


if __name__ == "__main__":
    main()
