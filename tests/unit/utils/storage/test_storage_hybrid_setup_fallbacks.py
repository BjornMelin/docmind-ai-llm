"""Hybrid setup fallback tests for storage helpers.

Covers ImportError paths in setup_hybrid_collection (sync/async), ensuring
fallback to dense-only store when hybrid requires optional dependencies.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_setup_hybrid_collection_importerror_fallback(monkeypatch):
    """When hybrid constructor raises ImportError, fallback uses dense-only."""
    mod = importlib.import_module("src.utils.storage")

    class _Client:
        def __init__(self):
            self.created = False

        def collection_exists(self, _name):
            return False

        def create_collection(self, **_):
            self.created = True

    class _Store:
        def __init__(self, client, collection_name, enable_hybrid, batch_size):
            # Hybrid attempt fails; dense fallback succeeds
            if enable_hybrid:
                raise ImportError("fastembed missing")
            self.args = (client, collection_name, enable_hybrid, batch_size)
            self.enable_hybrid = enable_hybrid

    monkeypatch.setattr(mod, "QdrantVectorStore", _Store)
    c = _Client()
    out = mod.setup_hybrid_collection(c, "col")
    assert isinstance(out, _Store)
    assert out.enable_hybrid is False
    assert c.created is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_setup_hybrid_collection_async_importerror_fallback(monkeypatch):
    """Async variant also falls back to dense-only on ImportError."""
    mod = importlib.import_module("src.utils.storage")

    class _AClient:
        async def collection_exists(self, _name):
            return False

        async def create_collection(self, **_):
            return None

        async def delete_collection(self, _):
            return None

    class _SClient:
        def __init__(self, **_):
            pass

    class _Store:
        def __init__(self, client, collection_name, enable_hybrid, batch_size):
            if enable_hybrid:
                raise ImportError("fastembed missing")
            self.args = (client, collection_name, enable_hybrid, batch_size)
            self.enable_hybrid = enable_hybrid

    monkeypatch.setattr(mod, "QdrantClient", _SClient)
    monkeypatch.setattr(mod, "QdrantVectorStore", _Store)
    out = await mod.setup_hybrid_collection_async(_AClient(), "col")
    assert isinstance(out, _Store)
    assert out.enable_hybrid is False
