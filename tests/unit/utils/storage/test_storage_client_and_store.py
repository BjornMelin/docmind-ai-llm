"""Unit tests for storage client config and vector store creation.

Targets:
- get_client_config returns expected keys and types
- create_vector_store ensures the named-vector schema in dense-only mode
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval import vector_contract
from tests.fixtures.test_settings import create_test_settings


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
def test_get_client_config_defends_against_mutated_remote_qdrant() -> None:
    """A post-validation mutation cannot bypass the network policy."""
    from src.utils.storage import get_client_config

    cfg = create_test_settings()
    cfg.database.qdrant_url = "https://qdrant.example.com"

    with pytest.raises(ValueError, match="endpoint security policy"):
        get_client_config(cfg)


@pytest.mark.unit
def test_create_vector_store_dense_only_ensures_named_schema(monkeypatch) -> None:
    """Dense-only stores still require the named dense-vector collection schema."""
    from src.utils import storage as storage_mod

    # Stub client and store to avoid network and external deps
    monkeypatch.setattr(storage_mod, "QdrantClient", MagicMock())
    monkeypatch.setattr(storage_mod, "AsyncQdrantClient", MagicMock())

    class _DummyStore:
        def __init__(
            self,
            client,
            aclient,
            collection_name,
            enable_hybrid,
            batch_size,
            dense_vector_name,
            sparse_vector_name,
            sparse_doc_fn,
            sparse_query_fn,
        ):
            self.client = client
            self._aclient = aclient
            self.collection_name = collection_name
            self.enable_hybrid = enable_hybrid
            self.batch_size = batch_size
            self.dense_vector_name = dense_vector_name
            self.sparse_vector_name = sparse_vector_name
            self.sparse_doc_fn = sparse_doc_fn
            self.sparse_query_fn = sparse_query_fn

    monkeypatch.setattr(storage_mod, "QdrantVectorStore", _DummyStore)

    calls = {"ensure": 0}

    def _count_ensure(*_args, **_kwargs):
        calls["ensure"] += 1
        return type("_Compatibility", (), {"compatible": True})()

    monkeypatch.setattr(storage_mod, "ensure_hybrid_collection", _count_ensure)

    store = storage_mod.create_vector_store("col", enable_hybrid=False)
    assert isinstance(store, _DummyStore)
    assert store.enable_hybrid is False
    assert store.dense_vector_name == vector_contract.DENSE_VECTOR_NAME
    assert store.sparse_vector_name == vector_contract.SPARSE_VECTOR_NAME
    assert calls["ensure"] == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory_name", ["create_vector_store", "connect_vector_store"]
)
def test_vector_store_constructor_failure_closes_sync_and_async_clients(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
) -> None:
    """A failed canonical construction releases both clients without fallback."""
    from src.utils import storage as storage_mod

    sync_client = MagicMock()

    class _AsyncClient:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    async_client = _AsyncClient()
    monkeypatch.setattr(storage_mod, "QdrantClient", lambda **_kwargs: sync_client)
    monkeypatch.setattr(
        storage_mod, "AsyncQdrantClient", lambda **_kwargs: async_client
    )
    compatibility = storage_mod.CollectionCompatibilityResult(
        compatible=True,
        action="unchanged",
        reason="compatible",
    )
    monkeypatch.setattr(
        storage_mod,
        "ensure_hybrid_collection",
        lambda *_args, **_kwargs: compatibility,
    )
    monkeypatch.setattr(
        storage_mod,
        "check_hybrid_collection",
        lambda *_args, **_kwargs: compatibility,
    )
    constructor = MagicMock(side_effect=RuntimeError("construction failed"))
    monkeypatch.setattr(storage_mod, "QdrantVectorStore", constructor)

    factory = getattr(storage_mod, factory_name)
    with pytest.raises(RuntimeError, match="construction failed"):
        factory("col", enable_hybrid=True)

    assert constructor.call_count == 1
    sync_client.close.assert_called_once_with()
    assert async_client.closed is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory_name", ["create_vector_store", "connect_vector_store"]
)
def test_async_client_constructor_failure_closes_sync_client(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
) -> None:
    """Failure to allocate the async peer cannot leak the sync transport."""
    from src.utils import storage as storage_mod

    sync_client = MagicMock()
    monkeypatch.setattr(storage_mod, "QdrantClient", lambda **_kwargs: sync_client)

    def _fail_async(**_kwargs: object) -> None:
        raise RuntimeError("async constructor failed")

    monkeypatch.setattr(storage_mod, "AsyncQdrantClient", _fail_async)
    compatibility = storage_mod.CollectionCompatibilityResult(
        compatible=True,
        action="unchanged",
        reason="compatible",
    )
    monkeypatch.setattr(
        storage_mod,
        "ensure_hybrid_collection",
        lambda *_args, **_kwargs: compatibility,
    )
    monkeypatch.setattr(
        storage_mod,
        "check_hybrid_collection",
        lambda *_args, **_kwargs: compatibility,
    )

    factory = getattr(storage_mod, factory_name)
    with pytest.raises(RuntimeError, match="async constructor failed"):
        factory("col")

    sync_client.close.assert_called_once_with()
