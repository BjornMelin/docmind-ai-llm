"""Integration-lite test: import chat page and validate chunking helper.

While we cannot execute Streamlit streaming in this environment, this test
ensures the chat page module is importable and the chunked streaming helper
behaves deterministically, which the page uses for streaming fallback.
"""

from __future__ import annotations

import importlib


def test_chat_page_chunking_helper_streams_text() -> None:
    mod = importlib.import_module("src.pages.01_chat")
    chunks = list(mod._chunked_stream("abcdef", chunk_size=2))  # type: ignore[attr-defined]
    assert chunks == ["ab", "cd", "ef"]
