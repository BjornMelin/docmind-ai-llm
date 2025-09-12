"""Basic tests for router_tool happy path with fake router engine.

Covers extraction of response_text and selected_strategy metadata.
"""

from __future__ import annotations

import json

import pytest

from src.agents.tools.router_tool import router_tool


class _Resp:
    def __init__(self, text: str, strategy: str | None = None) -> None:
        self.response = text
        self.metadata = {"selector_result": strategy} if strategy else {}


class _Router:
    def query(self, _q: str) -> _Resp:  # pragma: no cover - trivial
        return _Resp("ok", strategy="semantic_search")


@pytest.mark.unit
def test_router_tool_happy_path() -> None:
    """Test router tool extracts response text and selected strategy correctly."""
    out = router_tool.func("q", {"router_engine": _Router()})  # type: ignore[attr-defined]
    data = json.loads(out)
    assert data["response_text"] == "ok"
    assert data.get("selected_strategy") == "semantic_search"


@pytest.mark.unit
def test_router_tool_missing_engine_returns_error() -> None:
    """Test router tool returns error when router engine is missing."""
    out = router_tool.func("q", {"tools_data": {}})  # type: ignore[attr-defined]
    data = json.loads(out)
    assert "error" in data
    assert "router_engine" in data["error"]
