"""Unit tests for the LlamaIndex ingestion pipeline wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

from src.models.processing import IngestionConfig, IngestionInput
from src.processing.ingestion_api import load_documents_from_inputs
from src.processing.ingestion_pipeline import (
    _resolve_embedding,
    build_ingestion_pipeline,
    ingest_documents,
    ingest_documents_sync,
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
    assert result.metadata["docstore_enabled"] is True
    assert cfg.docstore_path is not None
    assert cfg.docstore_path.exists()


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
    assert result.metadata["cache_db"] == "docmind.duckdb"
    assert (tmp_path / "docstore.json").exists()
    assert result.documents
    for doc in result.documents:
        meta = getattr(doc, "metadata", {}) or {}
        assert "source_path" not in meta


def test_ingest_documents_sync_wrapper(tmp_path: Path) -> None:
    cfg = IngestionConfig(cache_dir=tmp_path / "cache")
    inputs = [IngestionInput(document_id="doc", payload_bytes=b"inline text")]

    result = ingest_documents_sync(cfg, inputs, embedding=DummyEmbedding())
    assert not result.exports


@pytest.mark.asyncio
async def test_ingest_documents_sync_guard() -> None:
    cfg = IngestionConfig()
    inputs = [IngestionInput(document_id="doc", payload_bytes=b"inline text")]

    with pytest.raises(RuntimeError) as exc_info:
        ingest_documents_sync(cfg, inputs, embedding=DummyEmbedding())
    assert "await ingest_documents" in str(exc_info.value)


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


def test_build_ingestion_pipeline_without_embedding(tmp_path: Path) -> None:
    """Pipeline construction succeeds when no embedding is configured."""
    cfg = IngestionConfig(
        chunk_size=64,
        chunk_overlap=16,
        cache_dir=tmp_path / "cache",
    )

    pipeline, _cache_path, _docstore_path = build_ingestion_pipeline(
        cfg, embedding=None
    )

    # TokenTextSplitter is always present even without embeddings.
    assert pipeline.transformations
    assert all(component is not None for component in pipeline.transformations)
    assert not any(
        isinstance(component, DummyEmbedding) for component in pipeline.transformations
    )


def test_resolve_embedding_configures_llamaindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_embedding calls setup_llamaindex when Settings lacks a model."""
    from src.processing import ingestion_pipeline as module

    dummy = DummyEmbedding()
    call_state = {"get": 0, "force": False}

    def fake_get_settings_embed_model() -> DummyEmbedding | None:  # pragma: no cover
        call_state["get"] += 1
        return dummy if call_state["get"] > 1 else None

    def fake_setup_llamaindex(*, force_embed: bool = False) -> None:  # pragma: no cover
        call_state["force"] = force_embed

    monkeypatch.setattr(
        module,
        "get_settings_embed_model",
        fake_get_settings_embed_model,
    )
    monkeypatch.setattr(module, "setup_llamaindex", fake_setup_llamaindex)

    resolved = _resolve_embedding(None)

    assert call_state == {"get": 2, "force": True}
    assert resolved is dummy


@pytest.mark.asyncio
async def test_ingest_documents_without_embedding_warns_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ingest_documents proceeds when embeddings remain unavailable."""
    from src.processing import ingestion_pipeline as module

    call_state: dict[str, Any] = {}

    async def _fake_arun(documents):  # type: ignore[no-untyped-def]
        return [
            {"doc_id": doc.doc_id, "text": getattr(doc, "text", "")}
            for doc in documents
        ]

    class _Pipeline:  # pragma: no cover - simple stub
        def __init__(self) -> None:
            self.docstore = SimpleNamespace(persist=lambda path: None)
            self.transformations: list[Any] = []

        async def arun(self, documents):  # type: ignore[no-untyped-def]
            return await _fake_arun(documents)

    def _fake_build(cfg, embedding):  # type: ignore[no-untyped-def]
        call_state["embedding"] = embedding
        pipeline = _Pipeline()
        return pipeline, tmp_path / "cache.duckdb", tmp_path / "docstore.json"

    monkeypatch.setattr(module, "build_ingestion_pipeline", _fake_build)
    monkeypatch.setattr(module, "get_settings_embed_model", lambda: None)
    monkeypatch.setattr(module, "setup_llamaindex", lambda **_: None)
    warnings: list[str] = []

    def _warn(msg: str, *args: Any, **kwargs: Any) -> None:
        warnings.append(str(msg))

    monkeypatch.setattr(module.logger, "warning", _warn, raising=False)

    cfg = IngestionConfig(
        cache_dir=tmp_path / "cache", docstore_path=tmp_path / "docstore.json"
    )
    inputs = [IngestionInput(document_id="doc", payload_bytes=b"payload")]

    result = await module.ingest_documents(cfg, inputs, embedding=None)

    assert call_state["embedding"] is None
    assert result.nodes == [{"doc_id": "doc", "text": "payload"}]
    assert any("No embedding model configured" in msg for msg in warnings)


@pytest.mark.asyncio
async def test_document_from_input_falls_back_on_type_error(tmp_path: Path) -> None:
    """TypeError from UnstructuredReader triggers text fallback path."""
    sample = tmp_path / "sample.txt"
    sample.write_text("Fallback content", encoding="utf-8")

    class ExplodingReader:
        def load_data(self, *args, **kwargs):  # pragma: no cover - simple stub
            raise TypeError("unexpected signature")

    item = IngestionInput(document_id="doc-1", source_path=sample)
    docs = await load_documents_from_inputs([item], reader=ExplodingReader())

    assert len(docs) == 1
    assert docs[0].doc_id == "doc-1"
    assert docs[0].text == "Fallback content"


def test_page_image_exports_builds_metadata(monkeypatch, tmp_path: Path) -> None:
    from src.processing import ingestion_pipeline as module

    entries = [
        {
            "page": 1,
            "image_path": str(tmp_path / "sample.webp"),
            "phash": "abc",
        },
        {
            "page": 2,
            "image_path": str(tmp_path / "sample.jpg.enc"),
            "phash": "def",
        },
    ]
    monkeypatch.setattr(module, "save_pdf_page_images", lambda *args, **kwargs: entries)

    pdf = tmp_path / "doc.pdf"
    pdf.write_text("pdf", encoding="utf-8")
    cfg = IngestionConfig(cache_dir=tmp_path)

    exports = module._page_image_exports(pdf, cfg, encrypt_override=False)

    assert exports[0].content_type == "image/webp"
    assert exports[1].content_type == "image/jpeg"


@pytest.mark.asyncio
async def test_load_documents_uses_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.processing import ingestion_pipeline as module

    class DummyReader:
        def load_data(  # type: ignore[no-untyped-def]
            self, *, file: Path, unstructured_kwargs: dict[str, str]
        ):
            _ = unstructured_kwargs
            return [SimpleNamespace(text="doc", doc_id="doc-1", metadata={})]

    from src.processing import ingestion_api as api

    monkeypatch.setattr(api, "_default_unstructured_reader", lambda: DummyReader())
    monkeypatch.setattr(
        module, "_page_image_exports", lambda *args, **kwargs: ["export"]
    )

    sample = tmp_path / "sample.pdf"
    sample.write_text("content", encoding="utf-8")
    cfg = IngestionConfig(cache_dir=tmp_path)
    inputs = [
        IngestionInput(document_id="doc-1", source_path=sample, encrypt_images=True)
    ]

    docs, exports = await module._load_documents(cfg, inputs)
    assert docs[0].metadata["document_id"] == "doc-1"
    assert exports == ["export"]
