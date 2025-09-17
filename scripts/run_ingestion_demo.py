"""Demonstrate the LlamaIndex-based ingestion pipeline with sample inputs.

This script is intentionally lightweight so developers can run a smoke test
without external dependencies. It processes a trivial in-memory document and
prints a summary of the resulting manifest and nodes.
"""

from __future__ import annotations

from pathlib import Path

from src.models.processing import IngestionConfig, IngestionInput
from src.processing.ingestion_pipeline import ingest_documents_sync


def main() -> None:
    """Run the ingestion pipeline against a synthetic payload."""
    cache_dir = Path("/tmp/docmind-ingestion-demo")
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = IngestionConfig(cache_dir=cache_dir)
    payload = IngestionInput(
        document_id="demo-doc",
        payload_bytes=b"DocMind ingestion demo",
        metadata={"source": "demo"},
    )
    result = ingest_documents_sync(config, [payload])

    print("=== Ingestion Demo Summary ===")
    print(f"Nodes generated : {len(result.nodes)}")
    print(f"Exports emitted : {len(result.exports)}")
    print(f"Duration (ms)   : {result.duration_ms:.2f}")
    print("Manifest summary:")
    print(f"  corpus_hash : {result.manifest.corpus_hash}")
    print(f"  config_hash : {result.manifest.config_hash}")
    print(f"  payloads    : {result.manifest.payload_count}")


if __name__ == "__main__":
    main()
