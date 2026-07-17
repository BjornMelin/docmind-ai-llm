"""Ensure create_vector_store enforces the canonical hybrid schema."""

from __future__ import annotations

import importlib

import pytest

from src.retrieval import vector_contract


@pytest.mark.unit
def test_create_vector_store_calls_ensure(monkeypatch):
    """Test that create_vector_store checks the canonical hybrid schema."""
    mod = importlib.import_module("src.utils.storage")

    calls = {"ensure_hybrid": 0}

    # Stub client and store
    class _Client:
        def close(self):
            return None

    class _AsyncClient:
        async def close(self) -> None:
            return None

    class _Store:
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
            self.args = (
                client,
                aclient,
                collection_name,
                enable_hybrid,
                batch_size,
                dense_vector_name,
                sparse_vector_name,
                sparse_doc_fn,
                sparse_query_fn,
            )

    monkeypatch.setattr(mod, "QdrantClient", lambda **_: _Client())
    monkeypatch.setattr(mod, "AsyncQdrantClient", lambda **_: _AsyncClient())
    monkeypatch.setattr(mod, "QdrantVectorStore", _Store)

    # Count calls to the canonical schema helper.
    def _ensure(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        calls["ensure_hybrid"] += 1
        return mod.CollectionCompatibilityResult(True, "unchanged", "compatible", 1)

    monkeypatch.setattr(mod, "ensure_hybrid_collection", _ensure)
    out = mod.create_vector_store("col", enable_hybrid=True)
    assert isinstance(out, _Store)
    assert calls["ensure_hybrid"] == 1
    assert out.args[5:7] == (
        vector_contract.DENSE_VECTOR_NAME,
        vector_contract.SPARSE_VECTOR_NAME,
    )


@pytest.mark.unit
def test_create_vector_store_closes_client_when_schema_is_blocked(monkeypatch):
    """A blocked schema check must not leak the temporary Qdrant client."""
    mod = importlib.import_module("src.utils.storage")

    class _Client:
        closed = False

        def close(self):
            self.closed = True

    client = _Client()
    monkeypatch.setattr(mod, "QdrantClient", lambda **_: client)
    monkeypatch.setattr(
        mod,
        "ensure_hybrid_collection",
        lambda *_args, **_kwargs: mod.CollectionCompatibilityResult(
            False,
            "blocked",
            "text_dense_dimension_mismatch",
            4,
        ),
    )

    with pytest.raises(mod.QdrantCollectionIncompatibleError):
        mod.create_vector_store("col")

    assert client.closed is True


@pytest.mark.unit
@pytest.mark.parametrize("hybrid_error", [ImportError, RuntimeError])
def test_create_vector_store_closes_client_when_construction_fails(
    monkeypatch,
    hybrid_error,
):
    """Constructor failures must not leak the Qdrant client."""
    mod = importlib.import_module("src.utils.storage")

    class _Client:
        closed = False

        def close(self):
            self.closed = True

    class _AsyncClient:
        closed = False

        async def close(self) -> None:
            self.closed = True

    client = _Client()
    async_client = _AsyncClient()
    calls = 0
    vector_names = []

    def _store(**kwargs):
        nonlocal calls
        calls += 1
        vector_names.append((kwargs["dense_vector_name"], kwargs["sparse_vector_name"]))
        if kwargs["enable_hybrid"]:
            raise hybrid_error("store failed")
        raise RuntimeError("dense fallback failed")

    monkeypatch.setattr(mod, "QdrantClient", lambda **_: client)
    monkeypatch.setattr(mod, "AsyncQdrantClient", lambda **_: async_client)
    monkeypatch.setattr(mod, "QdrantVectorStore", _store)
    monkeypatch.setattr(
        mod,
        "ensure_hybrid_collection",
        lambda *_args, **_kwargs: mod.CollectionCompatibilityResult(
            True,
            "unchanged",
            "compatible",
            1,
        ),
    )

    with pytest.raises(RuntimeError):
        mod.create_vector_store("col")

    assert calls == 1
    assert (
        vector_names
        == [(vector_contract.DENSE_VECTOR_NAME, vector_contract.SPARSE_VECTOR_NAME)]
        * calls
    )
    assert client.closed is True
    assert async_client.closed is True
