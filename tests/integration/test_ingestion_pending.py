"""Integration sanity test for the ingestion pipeline."""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index")

from src.models.processing import IngestionConfig, IngestionInput
from src.processing.ingestion_pipeline import ingest_documents_sync


@pytest.mark.integration
def test_ingestion_pipeline_smoke(tmp_path):
    """Run a minimal ingestion run to ensure the pipeline entrypoint wires."""
    cfg = IngestionConfig(cache_dir=tmp_path / "cache", enable_observability=False)
    inputs = [
        IngestionInput(
            document_id="doc-1",
            payload_bytes=b"Hello DocMind",
            metadata={"source": "test"},
        )
    ]
    result = ingest_documents_sync(cfg, inputs)
    assert result.nodes, "Pipeline should emit at least one node"
    assert result.manifest.corpus_hash
