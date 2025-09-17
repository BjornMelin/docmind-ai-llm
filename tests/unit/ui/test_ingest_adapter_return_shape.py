"""Unit tests for ingest adapter minimal return shape.

Verifies that calling `ingest_files` with an empty file list returns the
expected dict with count=0 and pg_index=None without performing heavy work.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

from src.models.processing import IngestionResult, ManifestSummary
from src.ui.ingest_adapter import ingest_files


def test_ingest_files_empty_returns_zero_and_no_pg() -> None:
    """When no files are provided, returns count=0 and no pg_index."""
    out = ingest_files([], enable_graphrag=True)
    assert isinstance(out, dict)
    assert out.get("count") == 0
    assert out.get("pg_index") is None
    assert out.get("vector_index") is None
    assert out.get("manifest") is None


class _DummyFile:
    """Simple stand-in for Streamlit UploadedFile."""

    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._buffer = io.BytesIO(payload)

    def read(self) -> bytes:
        return self._buffer.read()

    def getbuffer(self) -> memoryview:
        return memoryview(self._buffer.getvalue())

    def seek(self, pos: int, whence: int = 0) -> None:
        self._buffer.seek(pos, whence)


def test_ingest_files_builds_vector_and_optional_graph(monkeypatch, tmp_path):
    """Ingestion adapter wires vector index and optional graph index."""
    from src.ui import _ingest_adapter_impl as adapter

    monkeypatch.setattr(adapter.settings, "data_dir", tmp_path)
    monkeypatch.setattr(adapter.settings, "cache_dir", tmp_path / "cache")
    monkeypatch.setattr(adapter.settings.processing, "encrypt_page_images", False)
    obs = adapter.settings.observability.model_copy(
        update={"enabled": False, "endpoint": None, "sampling_ratio": 1.0}
    )
    monkeypatch.setattr(adapter.settings, "observability", obs)
    monkeypatch.setattr(adapter.settings.database, "qdrant_collection", "test")
    monkeypatch.setattr(adapter.settings.database, "vector_store_type", "qdrant")
    monkeypatch.setattr(adapter.settings.retrieval, "enable_server_hybrid", False)
    monkeypatch.setattr(adapter.settings, "app_version", "1.0.0")

    fake_result = IngestionResult(
        nodes=[object()],
        documents=[SimpleNamespace()],
        manifest=ManifestSummary(
            corpus_hash="abc12345", config_hash="def67890", payload_count=1
        ),
        exports=[],
        metadata={},
        duration_ms=12.5,
    )

    monkeypatch.setattr(adapter, "ingest_documents_sync", lambda *_, **__: fake_result)
    monkeypatch.setattr(adapter, "create_vector_store", lambda *_, **__: "store")
    monkeypatch.setattr(
        adapter,
        "StorageContext",
        SimpleNamespace(from_defaults=lambda **_: SimpleNamespace()),
    )
    monkeypatch.setattr(
        adapter,
        "VectorStoreIndex",
        lambda nodes, storage_context, show_progress: "vector-index",
    )
    monkeypatch.setattr(
        adapter,
        "PropertyGraphIndex",
        SimpleNamespace(
            from_documents=lambda documents, show_progress=False: SimpleNamespace(
                property_graph_store=object()
            )
        ),
    )

    dummy = _DummyFile("doc.txt", b"hello world")
    out = ingest_files([dummy], enable_graphrag=True)

    assert out["count"] == 1
    assert out["vector_index"] == "vector-index"
    assert out["pg_index"] is not None
    assert out["manifest"]["corpus_hash"] == "abc12345"
    upload_files = list((tmp_path / "uploads").glob("**/*"))
    assert upload_files, "uploaded file should be saved to the uploads directory"
