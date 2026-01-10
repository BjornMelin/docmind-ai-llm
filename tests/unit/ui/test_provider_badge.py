"""Unit tests for provider badge component."""

from __future__ import annotations

import ast
import inspect

import pytest
import streamlit as st

import src.ui.components.provider_badge as provider_badge_module
from src.config.settings import DocMindSettings
from src.ui.components.provider_badge import provider_badge


def _provider_badge_source() -> str:
    return inspect.getsource(provider_badge_module)


def _is_true_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


@pytest.mark.unit
def test_provider_badge_does_not_use_unsafe_allow_html() -> None:
    source_text = _provider_badge_source()
    tree = ast.parse(source_text)
    unsafe_calls: list[ast.Call] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "markdown":
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "st":
            continue

        for kw in node.keywords:
            if kw.arg != "unsafe_allow_html":
                continue
            if kw.value is not None and _is_true_constant(kw.value):
                unsafe_calls.append(node)
                break

    assert not unsafe_calls, "Found st.markdown(..., unsafe_allow_html=True)"


@pytest.mark.unit
def test_provider_badge_uses_config_values(monkeypatch: pytest.MonkeyPatch) -> None:
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
