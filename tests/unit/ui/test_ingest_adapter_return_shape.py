"""Unit tests for ingest adapter minimal return shape.

Verifies that calling `ingest_files` with an empty file list returns the
expected dict with count=0 and pg_index=None without performing heavy work.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

from src.models.processing import IngestionResult, ManifestSummary
from src.ui import _ingest_adapter_impl as adapter
from src.ui.ingest_adapter import ingest_files


def test_ingest_files_empty_returns_zero_and_no_pg(monkeypatch) -> None:
    """When no files are provided, returns count=0 and no pg_index."""
    monkeypatch.setattr(adapter, "setup_llamaindex", lambda **_: None)
    monkeypatch.setattr(adapter, "get_settings_embed_model", lambda: object())
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
    monkeypatch.setattr(adapter, "setup_llamaindex", lambda **_: None)
    monkeypatch.setattr(adapter, "get_settings_embed_model", lambda: object())

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


def test_ingest_files_skips_vector_index_without_embedding(
    monkeypatch, tmp_path, caplog
) -> None:
    """When embeddings are unavailable the adapter skips vector index creation."""
    monkeypatch.setattr(adapter.settings, "data_dir", tmp_path)
    monkeypatch.setattr(adapter.settings, "cache_dir", tmp_path / "cache")
    monkeypatch.setattr(adapter.settings.processing, "encrypt_page_images", False)
    obs = adapter.settings.observability.model_copy(
        update={"enabled": False, "endpoint": None, "sampling_ratio": 1.0}
    )
    monkeypatch.setattr(adapter.settings, "observability", obs)
    monkeypatch.setattr(adapter.settings.database, "qdrant_collection", "test")
    monkeypatch.setattr(adapter.settings.database, "vector_store_type", "qdrant")
    monkeypatch.setattr(adapter.settings, "app_version", "1.0.0")
    monkeypatch.setattr(adapter, "setup_llamaindex", lambda **_: None)
    monkeypatch.setattr(adapter, "get_settings_embed_model", lambda: None)

    fake_result = IngestionResult(
        nodes=[object()],
        documents=[SimpleNamespace()],
        manifest=ManifestSummary(
            corpus_hash="hashhash", config_hash="cfgcfgcfg", payload_count=1
        ),
        exports=[],
        metadata={},
        duration_ms=5.0,
    )
    monkeypatch.setattr(adapter, "ingest_documents_sync", lambda *_, **__: fake_result)

    def _fail(*_, **__):  # pragma: no cover - should not be invoked
        raise AssertionError("vector index should be skipped without embeddings")

    monkeypatch.setattr(adapter, "create_vector_store", _fail)
    monkeypatch.setattr(adapter, "StorageContext", SimpleNamespace(from_defaults=_fail))
    monkeypatch.setattr(adapter, "VectorStoreIndex", _fail)
    monkeypatch.setattr(adapter, "PropertyGraphIndex", None)

    dummy = _DummyFile("doc.txt", b"payload")
    with caplog.at_level("WARNING"):
        out = ingest_files([dummy], enable_graphrag=False)

    assert out["count"] == 1
    assert out["vector_index"] is None
    assert out["pg_index"] is None
    assert out["manifest"]["corpus_hash"] == "hashhash"
    assert any("vector index" in record.message for record in caplog.records)


def test_ingest_files_rechecks_embedding_after_ingestion(monkeypatch, tmp_path, caplog):
    """Vector index builds when embedding becomes available during ingestion."""
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
    monkeypatch.setattr(adapter, "setup_llamaindex", lambda **_: None)

    embed_value = object()
    state = {"calls": 0}

    def _embed_getter():
        state["calls"] += 1
        return None if state["calls"] == 1 else embed_value

    monkeypatch.setattr(adapter, "get_settings_embed_model", _embed_getter)

    fake_result = IngestionResult(
        nodes=[object()],
        documents=[SimpleNamespace()],
        manifest=ManifestSummary(
            corpus_hash="recheck-hash", config_hash="cfgcfgcfg", payload_count=1
        ),
        exports=[],
        metadata={},
        duration_ms=8.0,
    )
    monkeypatch.setattr(adapter, "ingest_documents_sync", lambda *_, **__: fake_result)

    store_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(
        adapter,
        "create_vector_store",
        lambda *a, **k: store_calls.append((a, k)) or "store",
    )
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
    monkeypatch.setattr(adapter, "PropertyGraphIndex", None)

    dummy = _DummyFile("doc.txt", b"payload")
    with caplog.at_level("INFO"):
        out = ingest_files([dummy], enable_graphrag=False)

    assert out["vector_index"] == "vector-index"
    assert store_calls, (
        "vector store should be instantiated when embedding becomes available"
    )
    assert any(
        "Embedding configured during ingestion" in record.message
        for record in caplog.records
    )
