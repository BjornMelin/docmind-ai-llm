"""Unit tests for provider badge component."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config.settings import DocMindSettings
from src.ui.components.provider_badge import provider_badge


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Failed to locate repo root from {start}")


def test_provider_badge_has_no_unsafe_html_sink() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve())
    source = repo_root / "src" / "ui" / "components" / "provider_badge.py"
    assert source.exists(), f"Expected source file not found: {source}"
    assert "unsafe_allow_html" not in source.read_text(encoding="utf-8")


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
