"""Basic tests for router_tool happy path with fake router engine.

Covers extraction of response_text and selected_strategy metadata.
"""

from __future__ import annotations

import json

import pytest

from src.agents.tools.router_tool import (
    _log_router_event,
    _resolve_router_engine,
    router_tool,
)


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
    out = router_tool.invoke(
        {"query": "q", "state": {"tools_data": {"router_engine": _Router()}}},
    )
    data = json.loads(out)
    assert data["response_text"] == "ok"
    assert data.get("selected_strategy") == "semantic_search"


@pytest.mark.unit
def test_router_tool_missing_engine_returns_error() -> None:
    """Test router tool returns error when router engine is missing."""
    out = router_tool.invoke({"query": "q"})
    data = json.loads(out)
    assert "error" in data
    assert "router_engine" in data["error"]


@pytest.mark.unit
def test_router_tool_extracts_text_and_message() -> None:
    """router_tool extracts .text or .message when .response is missing."""

    class _TextResp:
        def __init__(self) -> None:
            self.text = "text"
            self.metadata = {}

    class _MessageResp:
        def __init__(self) -> None:
            self.message = "msg"
            self.metadata = {}

    class _TextRouter:
        def query(self, _q: str) -> _TextResp:
            return _TextResp()

    class _MessageRouter:
        def query(self, _q: str) -> _MessageResp:
            return _MessageResp()

    out_text = router_tool.invoke(
        {"query": "q", "state": {"tools_data": {"router_engine": _TextRouter()}}},
    )
    assert json.loads(out_text)["response_text"] == "text"

    out_msg = router_tool.invoke(
        {"query": "q", "state": {"tools_data": {"router_engine": _MessageRouter()}}},
    )
    assert json.loads(out_msg)["response_text"] == "msg"


@pytest.mark.unit
def test_router_tool_query_error_returns_error_type() -> None:
    """router_tool returns error_type when router query raises."""

    class _BadRouter:
        def query(self, _q: str) -> _Resp:
            raise ValueError("boom")

    out = router_tool.invoke(
        {"query": "q", "state": {"tools_data": {"router_engine": _BadRouter()}}},
    )
    data = json.loads(out)
    assert data["error_type"] == "ValueError"


@pytest.mark.unit
def test_resolve_router_engine_from_runtime_config() -> None:
    """_resolve_router_engine reads configurable.runtime fields."""

    class _Config:
        def __init__(self) -> None:
            self.configurable = {"runtime": {"router_engine": "engine"}}

    engine = _resolve_router_engine(None, runtime_ctx=None, runtime_cfg=_Config())
    assert engine == "engine"


@pytest.mark.unit
def test_log_router_event_adds_traversal_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """_log_router_event adds traversal depth for knowledge_graph."""
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.agents.tools.router_tool.log_jsonl", lambda ev: events.append(ev)
    )

    _log_router_event("knowledge_graph", {"timing_ms": 1.0})
    assert events
    assert events[0].get("traversal_depth") is not None
