"""Unit test for exact staleness tooltip copy in Chat page.

Ensures the tooltip string matches SPEC-014 acceptance text exactly.
Uses the same simple import_module pattern as tests/helpers/run_settings_page.py.
"""

from __future__ import annotations

import importlib


def test_staleness_tooltip_exact_copy():
    chat_page = importlib.import_module("src.pages.01_chat")
    expected = (
        "Snapshot is stale (content/config changed). Rebuild in Documents → "
        "Rebuild GraphRAG Snapshot."
    )
    assert getattr(chat_page, "STALE_TOOLTIP", None) == expected
