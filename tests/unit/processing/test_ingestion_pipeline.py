"""Unit tests for the LlamaIndex ingestion pipeline wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

from src.models.processing import IngestionConfig, IngestionInput
from src.processing.ingestion_pipeline import (
    _document_from_input,
    build_ingestion_pipeline,
    ingest_documents,
)


class DummyEmbedding(BaseEmbedding):
    """Deterministic embedding for tests."""

    def _get_text_embedding(self, text: str):  # type: ignore[override]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        return [float(value)]

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
    assert Path(result.metadata["docstore_path"]).exists()


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
    assert (tmp_path / "docstore.json").exists()


@pytest.mark.asyncio
async def test_ingest_documents_sync_wrapper(tmp_path: Path) -> None:
    cfg = IngestionConfig(cache_dir=tmp_path / "cache")
    inputs = [IngestionInput(document_id="doc", payload_bytes=b"inline text")]

    result = await ingest_documents(cfg, inputs, embedding=DummyEmbedding())
    assert not result.exports


def test_build_ingestion_pipeline_uses_cache_and_docstore(tmp_path: Path) -> None:
    """build_ingestion_pipeline returns configured cache and docstore paths."""
    cfg = IngestionConfig(
        chunk_size=64,
        chunk_overlap=16,
        cache_dir=tmp_path / "cache",
        docstore_path=tmp_path / "docstore.json",
    )

    pipeline, cache_path, docstore_path = build_ingestion_pipeline(
        cfg, embedding=DummyEmbedding()
    )

    assert cache_path == cfg.cache_dir / "docmind.duckdb"
    assert docstore_path == cfg.docstore_path
    assert pipeline.transformations  # TokenTextSplitter + optional components


def test_document_from_input_falls_back_on_type_error(tmp_path: Path) -> None:
    """TypeError from UnstructuredReader triggers text fallback path."""
    sample = tmp_path / "sample.txt"
    sample.write_text("Fallback content", encoding="utf-8")

    class ExplodingReader:
        def load_data(self, *args, **kwargs):  # pragma: no cover - simple stub
            raise TypeError("unexpected signature")

    item = IngestionInput(document_id="doc-1", source_path=sample)
    docs = _document_from_input(ExplodingReader(), item)

    assert len(docs) == 1
    assert docs[0].doc_id == "doc-1"
    assert docs[0].text == "Fallback content"
