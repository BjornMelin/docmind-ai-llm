"""Unit tests for Chat page helper functions (no Streamlit runtime).

Covers small pure helpers to raise coverage for the chat page.
"""

from __future__ import annotations

import importlib


def test_chunked_stream_splits_text():  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.pages.01_chat")
    chunks = list(mod._chunked_stream("abcdefghij", chunk_size=3))
    assert chunks == ["abc", "def", "ghi", "j"]


def test_get_settings_override_builds_from_session(monkeypatch):  # type: ignore[no-untyped-def]
    mod = importlib.import_module("src.pages.01_chat")

    class _FakeSession(dict):
        pass

    fake_state = _FakeSession()
    fake_state["router_engine"] = object()
    fake_state["vector_index"] = 1
    fake_state["hybrid_retriever"] = 2
    fake_state["graphrag_index"] = 3

    st = importlib.import_module("streamlit")
    monkeypatch.setattr(st, "session_state", fake_state, raising=False)

    overrides = mod._get_settings_override()
    assert overrides is not None
    assert set(overrides.keys()) == {"router_engine", "vector", "retriever", "kg"}
