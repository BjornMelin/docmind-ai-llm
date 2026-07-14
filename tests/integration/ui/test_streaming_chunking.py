"""Integration-lite coverage for the chat page's streaming chunk helper."""

from __future__ import annotations

import importlib


def test_chat_page_chunking_helper_streams_text() -> None:
    """The chat page splits text into deterministic chunks."""
    mod = importlib.import_module("src.pages.01_chat")
    chunks = list(mod._chunked_stream("abcdef", chunk_size=2))  # type: ignore[attr-defined]
    assert chunks == ["ab", "cd", "ef"]
