"""Minimal tests for retrieval embeddings error path to raise coverage.

Covers ImportError path for BGEM3Embedding when FlagEmbedding is unavailable.
"""

import importlib

import pytest


def test_bgem3_import_error(monkeypatch):
    mod = importlib.import_module("src.retrieval.embeddings")
    # Force unavailable dependency
    monkeypatch.setattr(mod, "BGEM3FlagModel", None, raising=False)

    with pytest.raises(ImportError):
        mod.BGEM3Embedding()  # type: ignore[call-arg]
