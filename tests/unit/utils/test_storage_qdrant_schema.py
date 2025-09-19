"""Unit tests for Qdrant hybrid schema ensure/patch helpers.

Mocks a Qdrant client to exercise create/update code paths.
"""

from __future__ import annotations

import importlib
from typing import Any


def test_ensure_hybrid_collection_creates_when_missing(monkeypatch):  # type: ignore[no-untyped-def]
    """Ensure ensure_hybrid_collection creates a new schema when absent."""
    smod = importlib.import_module("src.utils.storage")

    calls: dict[str, int] = {"create": 0}

    class _Client:
        """Stub client that triggers create_collection when missing."""

        def collection_exists(self, _name: str) -> bool:  # type: ignore[no-untyped-def]
            return False

        def create_collection(self, **_kwargs: Any):  # type: ignore[no-untyped-def]
            calls["create"] += 1

    smod.ensure_hybrid_collection(_Client(), "c", dense_dim=128)
    assert calls["create"] == 1


def test_ensure_hybrid_collection_updates_when_present(monkeypatch):  # type: ignore[no-untyped-def]
    """Ensure ensure_hybrid_collection updates modifiers for existing schema."""
    smod = importlib.import_module("src.utils.storage")

    calls: dict[str, int] = {"update": 0}

    class _Vec:
        """Stub vector config storing the dense dimension."""

        def __init__(self):
            self.size = 64

    class _Sparse:
        """Stub sparse config exposing a modifier attribute for updates."""

        def __init__(self):
            self.modifier = None  # force patch to IDF

    class _Params:
        """Container for vector and sparse vector configs returned by client."""

        def __init__(self):
            self.vectors = {"text-dense": _Vec()}
            self.sparse_vectors = {"text-sparse": _Sparse()}

    class _Info:
        """Wrapper exposing a config attribute like the real Qdrant client."""

        def __init__(self):
            self.config = type("_", (), {"params": _Params()})()

    class _Client:
        """Client stub that reports an existing collection and tracks updates."""

        def collection_exists(self, _name: str) -> bool:  # type: ignore[no-untyped-def]
            return True

        def get_collection(self, _n: str):  # type: ignore[no-untyped-def]
            return _Info()

        def update_collection(self, **_kwargs: Any):  # type: ignore[no-untyped-def]
            calls["update"] += 1

    smod.ensure_hybrid_collection(_Client(), "c", dense_dim=64)
    assert calls["update"] == 1
