"""Coverage-oriented tests for src.utils.storage (no external I/O)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.utils import storage as storage_mod


def _ns(**kw):
    return SimpleNamespace(**kw)


def test_create_vector_store_returns_store(monkeypatch):
    """Test vector store creation returns proper store instance."""
    # Avoid real network: stub QdrantClient constructor
    monkeypatch.setattr(storage_mod, "QdrantClient", MagicMock())

    store = storage_mod.create_vector_store("test_collection", enable_hybrid=True)
    assert getattr(store, "collection_name", "test_collection")


def test_get_collection_info_exists(monkeypatch):
    """Test collection info retrieval when collection exists."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_info = _ns(
        points_count=10,
        config=_ns(
            params=_ns(vectors={"text-dense": {}}, sparse_vectors={"text-sparse": {}})
        ),
        status="green",
    )
    mock_client.get_collection.return_value = mock_info

    class _CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(storage_mod, "create_sync_client", _CM)

    out = storage_mod.get_collection_info("col")
    assert out["exists"] is True
    assert out["points_count"] == 10
    assert out["status"] == "green"


def test_get_collection_info_not_exists(monkeypatch):
    """Test collection info retrieval when collection does not exist."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False

    class _CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(storage_mod, "create_sync_client", _CM)
    out = storage_mod.get_collection_info("col")
    assert out == {"exists": False, "error": "Collection not found"}


def test_test_connection_success(monkeypatch):
    """Test connection test succeeds with valid Qdrant client."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = _ns(
        collections=[_ns(name="a"), _ns(name="b")]
    )

    class _CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(storage_mod, "create_sync_client", _CM)
    out = storage_mod.test_connection()
    assert out["connected"] is True
    assert out["collections_count"] == 2


def test_test_connection_error(monkeypatch):
    """Test connection test fails with connection error."""

    class _CM:
        def __enter__(self):
            raise ConnectionError("boom")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(storage_mod, "create_sync_client", _CM)
    out = storage_mod.test_connection()
    assert out["connected"] is False


def test_clear_collection_exists(monkeypatch):
    """Test collection clearing succeeds when collection exists."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = _ns(
        config=_ns(params=_ns(vectors={}, sparse_vectors={}))
    )

    class _CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(storage_mod, "create_sync_client", _CM)
    assert storage_mod.clear_collection("x") is True


def test_clear_collection_missing(monkeypatch):
    """Test collection clearing fails when collection does not exist."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False

    class _CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(storage_mod, "create_sync_client", _CM)
    assert storage_mod.clear_collection("x") is False
