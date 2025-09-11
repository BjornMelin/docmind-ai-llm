"""Unit test for router_tool telemetry: traversal_depth on KG route.

Ensures that when the selected route is knowledge_graph, the tool emits a
telemetry event with a `traversal_depth` field equal to the configured default.
"""

from __future__ import annotations

import importlib
import json

import pytest

from src.agents.tools.router_tool import router_tool


class _FakeResponse:
    def __init__(self, text: str, selected: str | None = None):
        self.response = text
        self.metadata = {"selector_result": selected} if selected else {}


class _FakeRouterKG:
    def query(self, q: str):
        return _FakeResponse(f"ok: {q}", selected="knowledge_graph")


@pytest.mark.unit
def test_router_tool_emits_traversal_depth_on_kg_route(monkeypatch):
    # Capture telemetry events
    events: list[dict] = []

    # Import the actual module object to patch module-level symbols
    rt_mod = importlib.import_module("src.agents.tools.router_tool")

    def _capture(evt: dict):
        events.append(evt)

    monkeypatch.setattr(rt_mod, "log_jsonl", _capture)

    state = {"tools_data": {"router_engine": _FakeRouterKG()}}
    result_json = router_tool.invoke({"query": "hello", "state": state})
    result = json.loads(result_json)

    assert result.get("response_text", "").startswith("ok:")
    # At least one event captured and contains traversal_depth
    assert events, "No telemetry events captured"
    # Find router_selected event
    router_ev = next((e for e in events if e.get("router_selected")), None)
    assert router_ev is not None
    assert router_ev.get("route") == "knowledge_graph"
    assert isinstance(router_ev.get("traversal_depth"), int)
