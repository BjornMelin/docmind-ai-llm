"""Additional error-path tests for storage helpers."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_get_collection_info_exception(monkeypatch):
    mod = importlib.import_module("src.utils.storage")

    class _CM:
        def __enter__(self):
            raise ConnectionError("boom")

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(mod, "create_sync_client", lambda: _CM())
    out = mod.get_collection_info("col")
    assert out["exists"] is False
    assert "error" in out


@pytest.mark.unit
def test_clear_collection_exception(monkeypatch):
    mod = importlib.import_module("src.utils.storage")

    class _Client:
        def collection_exists(self, _name):
            return True

        def get_collection(self, _):
            raise ConnectionError("oops")

        def close(self):
            return None

    class _CM:
        def __enter__(self):
            return _Client()

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(mod, "create_sync_client", lambda: _CM())
    assert mod.clear_collection("col") is False
