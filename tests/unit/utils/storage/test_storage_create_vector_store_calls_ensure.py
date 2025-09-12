"""Ensure create_vector_store enforces hybrid schema and sparse modifier calls."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_create_vector_store_calls_ensure(monkeypatch):
    """Test that create_vector_store calls ensure functions for hybrid schema."""
    mod = importlib.import_module("src.utils.storage")

    calls = {"ensure_hybrid": 0, "ensure_idf": 0}

    # Stub client and store
    class _Client:
        def close(self):
            return None

    class _Store:
        def __init__(self, client, collection_name, enable_hybrid, batch_size):
            self.args = (client, collection_name, enable_hybrid, batch_size)

    monkeypatch.setattr(mod, "QdrantClient", lambda **_: _Client())
    monkeypatch.setattr(mod, "QdrantVectorStore", _Store)

    # Count calls to ensure helpers
    monkeypatch.setattr(
        mod,
        "ensure_hybrid_collection",
        lambda *_, **__: calls.__setitem__("ensure_hybrid", calls["ensure_hybrid"] + 1),
    )
    monkeypatch.setattr(
        mod,
        "ensure_sparse_idf_modifier",
        lambda *_, **__: calls.__setitem__("ensure_idf", calls["ensure_idf"] + 1),
    )

    out = mod.create_vector_store("col", enable_hybrid=True)
    assert isinstance(out, _Store)
    # Called once each
    assert calls["ensure_hybrid"] == 1
    # IDF modifier ensure is best-effort; expect it attempts once
    assert calls["ensure_idf"] >= 1
