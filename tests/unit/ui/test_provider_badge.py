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

    settings = DocMindSettings()
    provider_badge(settings)

    assert any(
        isinstance(entry, tuple) and entry[0] == f"Provider: {settings.llm_backend}"
        for entry in calls["badge"]
    )
