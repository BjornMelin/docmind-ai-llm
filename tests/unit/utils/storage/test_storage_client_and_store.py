"""Unit tests for storage client config and vector store creation.

Targets:
- get_client_config returns expected keys and types
- create_vector_store ensures the named-vector schema in dense-only mode
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_get_client_config_keys_and_types() -> None:
    """get_client_config exposes url/timeout/prefer_grpc with proper types."""
    from src.utils.storage import get_client_config

    cfg = get_client_config()
    assert set(cfg.keys()) >= {"url", "timeout", "prefer_grpc"}
    assert isinstance(cfg["url"], str)
    assert isinstance(cfg["timeout"], (int, float))
    assert isinstance(cfg["prefer_grpc"], bool)


@pytest.mark.unit
def test_create_vector_store_dense_only_ensures_named_schema(monkeypatch) -> None:
    """Dense-only stores still require the named dense-vector collection schema."""
    from src.utils import storage as storage_mod

    # Stub client and store to avoid network and external deps
    monkeypatch.setattr(storage_mod, "QdrantClient", MagicMock())

    class _DummyStore:
        def __init__(
            self,
            client,
            collection_name,
            enable_hybrid,
            batch_size,
            dense_vector_name,
            sparse_vector_name,
        ):
            self.client = client
            self.collection_name = collection_name
            self.enable_hybrid = enable_hybrid
            self.batch_size = batch_size
            self.dense_vector_name = dense_vector_name
            self.sparse_vector_name = sparse_vector_name

    monkeypatch.setattr(storage_mod, "QdrantVectorStore", _DummyStore)

    calls = {"ensure": 0}

    def _count_ensure(*_args, **_kwargs):
        calls["ensure"] += 1
        return type("_Compatibility", (), {"compatible": True})()

    monkeypatch.setattr(storage_mod, "ensure_hybrid_collection", _count_ensure)

    store = storage_mod.create_vector_store("col", enable_hybrid=False)
    assert isinstance(store, _DummyStore)
    assert store.enable_hybrid is False
    assert store.dense_vector_name == storage_mod.DENSE_VECTOR_NAME
    assert store.sparse_vector_name == storage_mod.SPARSE_VECTOR_NAME
    assert calls["ensure"] == 1
