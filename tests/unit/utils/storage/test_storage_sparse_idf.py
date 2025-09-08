"""Tests for ensuring sparse modifier set to IDF in Qdrant collections."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.utils.storage import ensure_sparse_idf_modifier


def _info_with_modifier(modifier):
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                sparse_vectors={"text-sparse": SimpleNamespace(modifier=modifier)}
            )
        )
    )


def test_ensure_sparse_idf_modifier_updates_when_not_idf(monkeypatch):
    client = MagicMock()
    client.get_collection.return_value = _info_with_modifier(modifier="OTHER")

    ensure_sparse_idf_modifier(client, "col")

    assert client.update_collection.called
    args, kwargs = client.update_collection.call_args
    assert kwargs["collection_name"] == "col"
    assert "sparse_vectors_config" in kwargs
    assert "text-sparse" in kwargs["sparse_vectors_config"]


def test_ensure_sparse_idf_modifier_skips_when_already_idf(monkeypatch):
    from qdrant_client import models as qmodels

    client = MagicMock()
    client.get_collection.return_value = _info_with_modifier(
        modifier=qmodels.Modifier.IDF
    )

    ensure_sparse_idf_modifier(client, "col")
    client.update_collection.assert_not_called()
