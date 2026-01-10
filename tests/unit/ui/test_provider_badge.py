"""Unit tests for provider badge component."""

from __future__ import annotations

import streamlit as st

from src.config.settings import DocMindSettings
from src.ui.components.provider_badge import provider_badge


def test_provider_badge_uses_config_values(monkeypatch) -> None:
    calls: dict[str, list[object]] = {"badge": [], "caption": []}

    def _badge(message: str, **kwargs: object) -> None:
        calls["badge"].append((message, kwargs))

    def _caption(message: str) -> None:
        calls["caption"].append(message)

    monkeypatch.setattr(st, "badge", _badge)
    monkeypatch.setattr(st, "caption", _caption)

    provider_badge(DocMindSettings())

    assert any(
        isinstance(entry, tuple) and "Provider:" in entry[0] for entry in calls["badge"]
    )
