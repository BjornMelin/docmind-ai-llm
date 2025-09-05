"""Unit tests for `router_tool` JSON contract and behavior.

Validates success path with selector metadata, missing engine error path,
and robustness of response_text extraction.
"""

from __future__ import annotations

import json

import pytest

from src.agents.tools import router_tool


class _FakeResponse:
    def __init__(self, text: str, selected: str | None = None):
        """Store response text and optional selector metadata."""
        self.response = text
        self.metadata = {"selector_result": selected} if selected else {}


class _FakeRouter:
    def __init__(self, selected: str | None = None):
        """Initialize with an optional selected strategy."""
        self._selected = selected

    def query(self, q: str):
        """Return a fake response object for the query."""
        return _FakeResponse(f"ok: {q}", selected=self._selected)


@pytest.mark.unit
class TestRouterTool:
    """Contract and behavior tests for `router_tool`."""

    def test_router_tool_happy_path_with_strategy(self):
        """Captures selected_strategy when selector metadata is present."""
        engine = _FakeRouter(selected="hybrid_search")
        state = {"tools_data": {"router_engine": engine}}

        result_json = router_tool.invoke({"query": "hello", "state": state})
        result = json.loads(result_json)

        assert result.get("response_text", "").startswith("ok:")
        assert result.get("selected_strategy") == "hybrid_search"
        assert result.get("hybrid_used") is True
        assert isinstance(result.get("timing_ms"), (int | float))

    def test_router_tool_missing_engine(self):
        """Returns error JSON when router_engine is missing."""
        result_json = router_tool.invoke({"query": "hello"})
        result = json.loads(result_json)
        assert "error" in result

    def test_router_tool_no_metadata(self):
        """Still returns response_text and timing when no selector metadata."""
        engine = _FakeRouter(selected=None)
        state = {"tools_data": {"router_engine": engine}}
        result_json = router_tool.invoke({"query": "q", "state": state})
        result = json.loads(result_json)
        assert result.get("response_text", "").startswith("ok:")
        # selected_strategy may be absent; ensure no crash
        assert "error" not in result

    def test_router_tool_omits_strategy_fields_without_metadata(self):
        """When selector metadata is absent, strategy fields are omitted."""
        engine = _FakeRouter(selected=None)
        state = {"tools_data": {"router_engine": engine}}
        out = json.loads(router_tool.invoke({"query": "ping", "state": state}))
        assert "selected_strategy" not in out
        assert "multimodal_used" not in out
        assert "hybrid_used" not in out

    def test_router_tool_query_exception(self):
        """When router_engine.query raises, returns error JSON."""

        class _FailRouter:
            """Fake engine that raises on query for error-path testing."""

            def query(self, _q: str) -> None:
                """Simulate failing engine call."""
                raise RuntimeError("boom")

        state = {"tools_data": {"router_engine": _FailRouter()}}
        result_json = router_tool.invoke({"query": "hello", "state": state})
        result = json.loads(result_json)
        assert "error" in result
