"""Unit tests for ingest adapter minimal return shape.

Verifies that calling `ingest_files` with an empty file list returns the
expected dict with count=0 and pg_index=None without performing heavy work.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest
from llama_index.core.schema import TextNode

from src.models.processing import (
    CANONICAL_DOCUMENT_ID_KEY,
    IngestionInput,
    IngestionResult,
    ManifestSummary,
)
from src.processing.ingestion_api import load_documents_from_inputs
from src.ui import _ingest_adapter_impl as adapter
from src.ui.ingest_adapter import ingest_files


def _noop(**_: Any) -> None:
    """No-op stub for monkeypatching."""


def test_ingest_files_empty_returns_zero_and_no_pg(monkeypatch) -> None:
    """When no files are provided, returns count=0 and no pg_index."""
    monkeypatch.setattr(adapter, "setup_llamaindex", _noop)
    monkeypatch.setattr(adapter, "get_settings_embed_model", object)
    out = ingest_files([], enable_graphrag=True)
    assert isinstance(out, dict)
    assert out.get("count") == 0
    assert out.get("pg_index") is None
    assert out.get("vector_index") is None
    assert out.get("manifest") is None
    assert out.get("nlp_preview") is None


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


def test_ingest_inputs_rejects_duplicate_ids_before_setup(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        adapter,
        "setup_llamaindex",
        lambda **_kwargs: calls.append("embedding_setup"),
    )
    monkeypatch.setattr(
        adapter,
        "configure_observability",
        lambda *_args: calls.append("observability"),
    )
    inputs = [
        IngestionInput(document_id="doc-duplicate", payload_text="alpha"),
        IngestionInput(document_id="doc-duplicate", payload_text="beta"),
    ]

    with pytest.raises(ValueError, match=r"Duplicate document_id.*doc-duplicate"):
        adapter.ingest_inputs(inputs)

    assert calls == []


def _assert_skipped_document_vectors_untouched(
    monkeypatch: pytest.MonkeyPatch,
    *,
    inputs: list[IngestionInput],
    loaded_documents: list[Any],
    skipped_document_id: str,
) -> None:
    nodes = [
        TextNode(
            text=str(document.get_content()),
            metadata=dict(document.metadata),
        )
        for document in loaded_documents
    ]
    adapter._assign_stable_node_ids(nodes)
    previous_loaded_ids = {node.node_id for node in nodes}
    fake_result = IngestionResult(
        nodes=nodes,
        documents=loaded_documents,
        manifest=ManifestSummary(
            corpus_hash="replacement-scope-corpus",
            config_hash="replacement-scope-config",
            payload_count=len(nodes),
        ),
        exports=[],
        metadata={},
        duration_ms=1.0,
    )

    class _Client:
        def close(self) -> None:
            return None

    class _Store:
        collection_name = "documents"
        client = _Client()

        def __init__(self) -> None:
            self.deleted: list[str | int] = []

        def delete_nodes(self, *, node_ids: list[str | int]) -> None:
            self.deleted.extend(node_ids)

    store = _Store()
    replacement_scopes: list[set[str]] = []

    def _existing_ids(_store: object, document_ids: set[str]) -> set[str]:
        replacement_scopes.append(set(document_ids))
        existing = set(previous_loaded_ids)
        if skipped_document_id in document_ids:
            existing.add("skipped-document-point")
        return existing

    monkeypatch.setattr(adapter, "setup_llamaindex", _noop)
    monkeypatch.setattr(adapter, "get_settings_embed_model", object)
    monkeypatch.setattr(adapter, "embedding_allowed_for_ingestion", lambda _: True)
    monkeypatch.setattr(
        adapter,
        "configure_observability",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(adapter, "ingest_documents_sync", lambda *_, **__: fake_result)
    monkeypatch.setattr(adapter, "create_vector_store", lambda *_, **__: store)
    monkeypatch.setattr(adapter, "_existing_text_point_ids", _existing_ids)
    monkeypatch.setattr(
        adapter,
        "StorageContext",
        SimpleNamespace(from_defaults=lambda **_: SimpleNamespace()),
    )
    monkeypatch.setattr(
        adapter,
        "VectorStoreIndex",
        lambda *_args, **_kwargs: "vector-index",
    )

    result = adapter.ingest_inputs(inputs)

    loaded_ids = {
        str(document.metadata[CANONICAL_DOCUMENT_ID_KEY])
        for document in loaded_documents
    }
    assert result["vector_index"] == "vector-index"
    assert replacement_scopes == [loaded_ids]
    assert skipped_document_id not in replacement_scopes[0]
    assert store.deleted == []


@pytest.mark.asyncio
async def test_missing_source_is_excluded_from_vector_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    missing = IngestionInput(
        document_id="doc-missing",
        source_path=tmp_path / "missing.txt",
    )
    valid = IngestionInput(document_id="doc-valid", payload_text="valid content")
    inputs = [missing, valid]

    loaded_documents = await load_documents_from_inputs(inputs)

    _assert_skipped_document_vectors_untouched(
        monkeypatch,
        inputs=inputs,
        loaded_documents=loaded_documents,
        skipped_document_id=missing.document_id,
    )


@pytest.mark.asyncio
async def test_symlinked_source_is_excluded_from_vector_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("target content", encoding="utf-8")
    link = tmp_path / "source-link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    symlinked = IngestionInput(document_id="doc-symlinked", source_path=link)
    valid = IngestionInput(document_id="doc-valid", payload_text="valid content")
    inputs = [symlinked, valid]

    loaded_documents = await load_documents_from_inputs(inputs)

    _assert_skipped_document_vectors_untouched(
        monkeypatch,
        inputs=inputs,
        loaded_documents=loaded_documents,
        skipped_document_id=symlinked.document_id,
    )


def test_ingest_files_runs_duplicate_preflight_before_delegation(
    monkeypatch,
    tmp_path,
) -> None:
    digest = "a" * 64
    paths = iter((tmp_path / "first.txt", tmp_path / "second.txt"))
    monkeypatch.setattr(
        adapter,
        "save_uploaded_file",
        lambda _file: (next(paths), digest),
    )
    delegated = False

    def _delegate(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal delegated
        delegated = True
        return {}

    monkeypatch.setattr(adapter, "ingest_inputs", _delegate)

    with pytest.raises(ValueError, match=r"Duplicate document_id"):
        adapter.ingest_files(
            [_DummyFile("first.txt", b"same"), _DummyFile("second.txt", b"same")]
        )

    assert delegated is False


def test_ingest_files_builds_vector_and_optional_graph(monkeypatch, tmp_path):
    """Ingestion adapter wires vector index and optional graph index."""
    monkeypatch.setattr(adapter.settings, "data_dir", tmp_path)
    monkeypatch.setattr(adapter.settings.cache, "dir", tmp_path / "cache")
    monkeypatch.setattr(adapter.settings.processing, "encrypt_page_images", False)
    obs = adapter.settings.observability.model_copy(
        update={"enabled": False, "endpoint": None, "sampling_ratio": 1.0}
    )
    monkeypatch.setattr(adapter.settings, "observability", obs)
    monkeypatch.setattr(adapter.settings.database, "qdrant_collection", "test")
    monkeypatch.setattr(adapter.settings.database, "vector_store_type", "qdrant")
    monkeypatch.setattr(adapter.settings.retrieval, "enable_server_hybrid", False)
    monkeypatch.setattr(adapter, "setup_llamaindex", _noop)
    monkeypatch.setattr(adapter, "get_settings_embed_model", object)

    fake_result = IngestionResult(
        nodes=[SimpleNamespace(node_id="node-1", metadata={})],
        documents=[SimpleNamespace(metadata={CANONICAL_DOCUMENT_ID_KEY: "doc-loaded"})],
        manifest=ManifestSummary(
            corpus_hash="abc12345", config_hash="def67890", payload_count=1
        ),
        exports=[],
        metadata={},
        duration_ms=12.5,
    )

    monkeypatch.setattr(adapter, "ingest_documents_sync", lambda *_, **__: fake_result)
    monkeypatch.setattr(adapter, "create_vector_store", lambda *_, **__: "store")
    monkeypatch.setattr(adapter, "_existing_text_point_ids", lambda *_: set())
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
    monkeypatch.setattr(adapter.settings.cache, "dir", tmp_path / "cache")
    monkeypatch.setattr(adapter.settings.processing, "encrypt_page_images", False)
    obs = adapter.settings.observability.model_copy(
        update={"enabled": False, "endpoint": None, "sampling_ratio": 1.0}
    )
    monkeypatch.setattr(adapter.settings, "observability", obs)
    monkeypatch.setattr(adapter.settings.database, "qdrant_collection", "test")
    monkeypatch.setattr(adapter.settings.database, "vector_store_type", "qdrant")
    monkeypatch.setattr(adapter, "setup_llamaindex", _noop)
    monkeypatch.setattr(adapter, "get_settings_embed_model", lambda: None)

    fake_result = IngestionResult(
        nodes=[SimpleNamespace(node_id="node-1", metadata={})],
        documents=[SimpleNamespace(metadata={CANONICAL_DOCUMENT_ID_KEY: "doc-loaded"})],
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


def test_ingest_files_skips_vector_index_when_embedding_blocked(
    monkeypatch, tmp_path, caplog
) -> None:
    """Endpoint-blocked embeddings do not trigger vector index creation."""
    monkeypatch.setattr(adapter.settings, "data_dir", tmp_path)
    monkeypatch.setattr(adapter.settings.cache, "dir", tmp_path / "cache")
    monkeypatch.setattr(adapter.settings.processing, "encrypt_page_images", False)
    obs = adapter.settings.observability.model_copy(
        update={"enabled": False, "endpoint": None, "sampling_ratio": 1.0}
    )
    monkeypatch.setattr(adapter.settings, "observability", obs)
    monkeypatch.setattr(adapter, "setup_llamaindex", _noop)
    endpoint_embedding = SimpleNamespace(api_base="https://api.openai.com/v1")
    monkeypatch.setattr(adapter, "get_settings_embed_model", lambda: endpoint_embedding)
    monkeypatch.setattr(
        adapter,
        "embedding_allowed_for_ingestion",
        lambda embedding: False,
    )

    fake_result = IngestionResult(
        nodes=[object()],
        documents=[SimpleNamespace(metadata={CANONICAL_DOCUMENT_ID_KEY: "doc-loaded"})],
        manifest=ManifestSummary(
            corpus_hash="hashhash", config_hash="cfgcfgcfg", payload_count=1
        ),
        exports=[],
        metadata={},
        duration_ms=5.0,
    )
    monkeypatch.setattr(adapter, "ingest_documents_sync", lambda *_, **__: fake_result)

    def _fail(*_, **__):  # pragma: no cover - should not be invoked
        raise AssertionError("vector index should be skipped when blocked")

    monkeypatch.setattr(adapter, "create_vector_store", _fail)
    monkeypatch.setattr(adapter, "StorageContext", SimpleNamespace(from_defaults=_fail))
    monkeypatch.setattr(adapter, "VectorStoreIndex", _fail)
    monkeypatch.setattr(adapter, "PropertyGraphIndex", None)

    dummy = _DummyFile("doc.txt", b"payload")
    with caplog.at_level("WARNING"):
        out = ingest_files([dummy], enable_graphrag=False)

    assert out["vector_index"] is None
    assert any(
        "blocked by endpoint policy" in record.message for record in caplog.records
    )


def test_ingest_files_rechecks_embedding_after_ingestion(monkeypatch, tmp_path, caplog):
    """Vector index builds when embedding becomes available during ingestion."""
    monkeypatch.setattr(adapter.settings, "data_dir", tmp_path)
    monkeypatch.setattr(adapter.settings.cache, "dir", tmp_path / "cache")
    monkeypatch.setattr(adapter.settings.processing, "encrypt_page_images", False)
    obs = adapter.settings.observability.model_copy(
        update={"enabled": False, "endpoint": None, "sampling_ratio": 1.0}
    )
    monkeypatch.setattr(adapter.settings, "observability", obs)
    monkeypatch.setattr(adapter.settings.database, "qdrant_collection", "test")
    monkeypatch.setattr(adapter.settings.database, "vector_store_type", "qdrant")
    monkeypatch.setattr(adapter.settings.retrieval, "enable_server_hybrid", False)
    monkeypatch.setattr(adapter, "setup_llamaindex", _noop)

    embed_value = object()
    state = {"calls": 0}

    def _embed_getter():
        state["calls"] += 1
        return None if state["calls"] == 1 else embed_value

    monkeypatch.setattr(adapter, "get_settings_embed_model", _embed_getter)

    fake_result = IngestionResult(
        nodes=[SimpleNamespace(node_id="node-1", metadata={})],
        documents=[SimpleNamespace(metadata={CANONICAL_DOCUMENT_ID_KEY: "doc-loaded"})],
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
    monkeypatch.setattr(adapter, "_existing_text_point_ids", lambda *_: set())
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


def test_build_vector_index_replaces_stale_document_points(monkeypatch) -> None:
    class _Point:
        id = "stale-point"

    class _Client:
        def scroll(self, **_kwargs):  # type: ignore[no-untyped-def]
            return [_Point()], None

        def close(self) -> None:
            return None

    class _Store:
        collection_name = "documents"
        client = _Client()

        def __init__(self) -> None:
            self.deleted: list[str | int] = []

        def delete_nodes(self, *, node_ids):  # type: ignore[no-untyped-def]
            self.deleted.extend(node_ids)

    store = _Store()
    node = TextNode(
        text="hello",
        metadata={CANONICAL_DOCUMENT_ID_KEY: "doc-1", "page_id": "page-1"},
    )
    monkeypatch.setattr(adapter, "create_vector_store", lambda *_, **__: store)
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

    first = adapter._build_vector_index([node], document_ids={"doc-1"})
    first_id = node.node_id
    second_node = TextNode(
        text="changed",
        metadata={CANONICAL_DOCUMENT_ID_KEY: "doc-1", "page_id": "page-1"},
    )
    adapter._assign_stable_node_ids([second_node])

    assert first == "vector-index"
    assert store.deleted == ["stale-point"]
    assert first_id == second_node.node_id


def test_build_vector_index_rejects_empty_replacement(monkeypatch) -> None:
    monkeypatch.setattr(
        adapter,
        "create_vector_store",
        lambda *_args, **_kwargs: pytest.fail(
            "vector store opened for an empty replacement"
        ),
    )

    with pytest.raises(ValueError, match="empty node set"):
        adapter._build_vector_index([], document_ids={"doc-1"})


def test_build_vector_index_surfaces_stale_deletion_failure(monkeypatch) -> None:
    class _Point:
        id = "stale-point"

    class _Client:
        closed = False

        def scroll(self, **_kwargs):  # type: ignore[no-untyped-def]
            return [_Point()], None

        def close(self) -> None:
            self.closed = True

    class _Store:
        collection_name = "documents"
        client = _Client()

        def delete_nodes(self, *, node_ids):  # type: ignore[no-untyped-def]
            assert node_ids == ["stale-point"]
            raise RuntimeError("delete failed")

    node = TextNode(
        text="replacement",
        metadata={CANONICAL_DOCUMENT_ID_KEY: "doc-1", "page_id": "page-1"},
    )
    store = _Store()
    monkeypatch.setattr(adapter, "create_vector_store", lambda *_args, **_kwargs: store)
    monkeypatch.setattr(
        adapter,
        "StorageContext",
        SimpleNamespace(from_defaults=lambda **_kwargs: SimpleNamespace()),
    )
    monkeypatch.setattr(
        adapter,
        "VectorStoreIndex",
        lambda *_args, **_kwargs: "upsert-succeeded",
    )

    with pytest.raises(RuntimeError, match="stale-point deletion failed") as raised:
        adapter._build_vector_index([node], document_ids={"doc-1"})

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert store.client.closed is True
