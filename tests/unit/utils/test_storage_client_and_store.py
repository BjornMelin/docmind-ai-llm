"""Unit tests for storage client config and vector store creation.

Targets:
- get_client_config returns expected keys and types
- create_vector_store with enable_hybrid=False does not call ensure helper
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
def test_create_vector_store_no_hybrid_skips_ensure(monkeypatch) -> None:
    """When enable_hybrid=False, ensure_hybrid_collection must not be called."""
    from src.utils import storage as storage_mod

    # Stub client and store to avoid network and external deps
    monkeypatch.setattr(storage_mod, "QdrantClient", MagicMock())

    class _DummyStore:
        def __init__(self, client, collection_name, enable_hybrid, batch_size):
            self.client = client
            self.collection_name = collection_name
            self.enable_hybrid = enable_hybrid
            self.batch_size = batch_size

    monkeypatch.setattr(storage_mod, "QdrantVectorStore", _DummyStore)

    calls = {"ensure": 0}

    def _count_ensure(*_args, **_kwargs):
        calls["ensure"] += 1

    monkeypatch.setattr(storage_mod, "ensure_hybrid_collection", _count_ensure)

    store = storage_mod.create_vector_store("col", enable_hybrid=False)
    assert isinstance(store, _DummyStore)
    assert store.enable_hybrid is False
    assert calls["ensure"] == 0
