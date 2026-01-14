"""Integration test for PDF page image exports + indexing wiring.

Validates the final-release multimodal ingestion flow without requiring a live
Qdrant instance or model downloads.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

from src.models.processing import IngestionConfig, IngestionInput
from src.processing.ingestion_pipeline import ingest_documents_sync


class DummyEmbedding(BaseEmbedding):
    """Deterministic embedding for pipeline construction in tests."""

    EMBEDDING_DIM: ClassVar[int] = 768

    def _get_text_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(b) / 255.0 for b in digest * (self.EMBEDDING_DIM // len(digest))]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._get_text_embedding(t) for t in texts]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._get_text_embedding(t) for t in texts]


def _skip_if_no_pymupdf() -> None:
    import importlib.util

    if importlib.util.find_spec("fitz") is None:
        pytest.skip("PyMuPDF not available")


def _make_pdf(path: Path) -> None:
    import fitz  # type: ignore

    doc = fitz.open()  # type: ignore[no-untyped-call]
    page = doc.new_page(width=200, height=100)  # type: ignore[attr-defined]
    page.insert_text((20, 50), "Hello multimodal")
    doc.save(str(path))
    doc.close()


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [1, 64])
def test_ingestion_pdf_images_exports_and_index_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    """Ingest a tiny PDF and assert page-image exports reach image indexer."""
    _skip_if_no_pymupdf()

    from src.config.settings import settings as app_settings

    monkeypatch.setattr(app_settings, "data_dir", tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_settings, "cache_dir", cache_dir)

    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path)

    doc_id = "doc-" + hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]

    cfg = IngestionConfig(
        chunk_size=128,
        chunk_overlap=16,
        cache_dir=tmp_path / "cache" / "ingestion",
        docstore_path=tmp_path / "cache" / "ingestion" / "docstore.json",
        enable_image_indexing=True,
        enable_image_encryption=False,
        image_index_batch_size=batch_size,
    )
    inputs = [IngestionInput(document_id=doc_id, source_path=pdf_path)]

    # Stub the heavy ingestion pipeline execution.
    class _Pipeline:  # pragma: no cover - test stub
        def __init__(self) -> None:
            self.docstore = SimpleNamespace(persist=lambda _: None)
            self.transformations = []

        async def arun(self, documents: Sequence[object]) -> list[object]:
            del documents
            return []

    monkeypatch.setattr(
        "src.processing.ingestion_pipeline.build_ingestion_pipeline",
        lambda *_a, **_k: (_Pipeline(), tmp_path / "cache.duckdb", None),  # pyright: ignore[unused-param]
    )
    monkeypatch.setattr(
        "src.processing.ingestion_pipeline._resolve_embedding",
        lambda *_a, **_k: None,  # pyright: ignore[unused-param]
    )

    captured: dict[str, object] = {"records": None, "batch_size": None}

    def _fake_index(
        *_a: object,
        records: Sequence[dict[str, Any]],
        batch_size: int = 0,
        **_k: object,  # pyright: ignore[unused-param]
    ) -> int:
        captured["records"] = list(records)
        captured["batch_size"] = batch_size
        return len(records)

    class _DummySiglip:  # pragma: no cover - avoid model init
        pass

    class _DummyClient:  # pragma: no cover - avoid network
        def __init__(self, **_cfg: object) -> None:
            self._cfg = _cfg

        def close(self) -> None:
            return

    monkeypatch.setattr("qdrant_client.QdrantClient", _DummyClient)
    monkeypatch.setattr("src.utils.siglip_adapter.SiglipEmbedding", _DummySiglip)
    monkeypatch.setattr(
        "src.retrieval.image_index.index_page_images_siglip", _fake_index
    )

    result = ingest_documents_sync(cfg, inputs, embedding=DummyEmbedding())

    assert result.exports, "Expected page-image exports for PDF"
    assert result.metadata.get("image_index.enabled") is True
    assert int(result.metadata.get("image_index.indexed", 0)) >= 1
    assert captured["records"] is not None
    assert captured["batch_size"] == batch_size
