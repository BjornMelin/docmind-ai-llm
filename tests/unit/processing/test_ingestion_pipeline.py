"""Unit tests for the LlamaIndex ingestion pipeline wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

from src.models.processing import IngestionConfig, IngestionInput
from src.processing.ingestion_pipeline import ingest_documents


class DummyEmbedding(BaseEmbedding):
    """Deterministic embedding for tests."""

    def _get_text_embedding(self, text: str):  # type: ignore[override]
        return [float(len(text) % 5)]

    async def _aget_text_embedding(self, text: str):  # type: ignore[override]
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str):  # type: ignore[override]
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str):  # type: ignore[override]
        return self._get_text_embedding(query)

    def _get_text_embeddings(self, texts):  # type: ignore[override]
        return [self._get_text_embedding(t) for t in texts]

    async def _aget_text_embeddings(self, texts):  # type: ignore[override]
        return [self._get_text_embedding(t) for t in texts]


@pytest.mark.asyncio
async def test_ingest_documents_with_bytes_payload(tmp_path: Path) -> None:
    cfg = IngestionConfig(
        chunk_size=64,
        chunk_overlap=16,
        cache_dir=tmp_path / "cache",
        docstore_path=tmp_path / "docstore.json",
    )
    inputs = [
        IngestionInput(document_id="doc-1", payload_bytes=b"DocMind ingestion test.")
    ]

    result = await ingest_documents(cfg, inputs, embedding=DummyEmbedding())

    assert result.duration_ms >= 0
    assert result.manifest.payload_count == len(result.nodes)
    assert result.metadata["document_count"] == 1


@pytest.mark.asyncio
async def test_ingest_documents_with_path(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("Library first ingestion pipeline")

    cfg = IngestionConfig(
        chunk_size=128,
        chunk_overlap=32,
        cache_dir=tmp_path / "cache",
        docstore_path=tmp_path / "docstore.json",
    )
    inputs = [
        IngestionInput(
            document_id="doc-file",
            source_path=source,
            metadata={"source_tag": "unit"},
        )
    ]

    result = await ingest_documents(cfg, inputs, embedding=DummyEmbedding())

    assert result.manifest.corpus_hash
    assert result.metadata["cache_path"].endswith("docmind.duckdb")


@pytest.mark.asyncio
async def test_ingest_documents_sync_wrapper(tmp_path: Path) -> None:
    cfg = IngestionConfig(cache_dir=tmp_path / "cache")
    inputs = [IngestionInput(document_id="doc", payload_bytes=b"inline text")]

    result = await ingest_documents(cfg, inputs, embedding=DummyEmbedding())
    assert not result.exports
