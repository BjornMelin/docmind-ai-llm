"""Tests for sparse IDF modifier enforcement in storage utilities."""

import pytest

pytestmark = pytest.mark.unit


def test_ensure_sparse_idf_modifier_updates_when_not_idf(monkeypatch):
    """Ensure modifier is updated to IDF when current modifier differs.

    Mocks a client that exposes a `get_collection` with a config showing a
    non-IDF sparse modifier and asserts that `update_collection` is invoked.
    """
    from types import SimpleNamespace

    from src.utils import storage as storage_mod

    called = {"update": False}

    class _SparseParam:
        def __init__(self, modifier):
            self.modifier = modifier

    class _Info:
        def __init__(self):
            self.config = SimpleNamespace(
                params=SimpleNamespace(
                    sparse_vectors={"text-sparse": _SparseParam(modifier="BM25")}
                )
            )

    class _Client:
        def get_collection(self, _):
            return _Info()

        def update_collection(self, **_):
            called["update"] = True

    storage_mod.ensure_sparse_idf_modifier(_Client(), "col")
    assert called["update"] is True
