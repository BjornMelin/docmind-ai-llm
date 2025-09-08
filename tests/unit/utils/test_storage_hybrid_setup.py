"""Tests for hybrid collection setup (sync path) in storage utilities.

Google-Style Docstrings:
    Ensures that when `recreate=True` and the collection exists, the setup
    deletes and re-creates the collection and returns a vector store instance.
"""

import pytest

pytestmark = pytest.mark.unit


def test_setup_hybrid_collection_recreate(monkeypatch):
    """Test that hybrid setup deletes and recreates the collection when needed.

    Patches the client and QdrantVectorStore to be simple stubs, verifies that
    the function performs delete and create operations and returns a store.
    """
    from types import SimpleNamespace

    from src.utils import storage as storage_mod

    class _Client:
        def __init__(self):
            self.deleted = False
            self.created = False

        def collection_exists(self, _):
            return True

        def delete_collection(self, _):
            self.deleted = True

        def create_collection(self, **_):
            self.created = True

        def get_collection(self, _):
            return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace()))

    class _Store:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(storage_mod, "QdrantVectorStore", _Store)

    client = _Client()
    out = storage_mod.setup_hybrid_collection(client, "col", recreate=True)
    assert isinstance(out, _Store)
    assert client.deleted is True
    assert client.created is True
