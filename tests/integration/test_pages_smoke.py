"""Smoke tests for new Streamlit pages."""

from __future__ import annotations

import importlib


def test_chat_page_importable() -> None:
    """Import chat page without executing top-level failures."""
    mod = importlib.import_module("src.pages.01_chat".replace(".", "."))
    assert mod is not None


def test_documents_page_importable() -> None:
    """Import documents page without executing top-level failures."""
    mod = importlib.import_module("src.pages.02_documents".replace(".", "."))
    assert mod is not None


def test_analytics_page_importable() -> None:
    """Import analytics page without executing top-level failures."""
    mod = importlib.import_module("src.pages.03_analytics".replace(".", "."))
    assert mod is not None
