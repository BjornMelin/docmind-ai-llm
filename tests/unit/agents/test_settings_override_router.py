"""Unit test to verify settings_override passes router_engine to tools_data.

This test patches internal methods of MultiAgentCoordinator to avoid heavy
dependencies and ensures that the router_engine override is preserved in the
initial state passed to the agent workflow.
"""
# pylint: disable=protected-access

from __future__ import annotations

from types import SimpleNamespace

from src.agents.coordinator import MultiAgentCoordinator


def test_settings_override_router_engine_passthrough() -> None:
    """Router engine passed via settings_override should reach tools_data."""
    coord = MultiAgentCoordinator()

    # Patch setup to skip heavy initialization
    coord._ensure_setup = lambda: True  # type: ignore[assignment]

    captured = {}

    def fake_run(initial_state, _thread_id):  # type: ignore[no-untyped-def]
        # MultiAgentState is a Pydantic model; access attributes directly
        captured["tools_data"] = dict(getattr(initial_state, "tools_data", {}) or {})
        # Return a minimal state compatible with _extract_response
        return {
            "messages": [{"content": "ok"}],
            "response": "answer",
            "coordination_time": 0.01,
        }

    def fake_extract(
        _state,
        _query,
        _start_time,
        _coordination_time,
    ):  # type: ignore[no-untyped-def]
        return SimpleNamespace(content="answer")

    coord._run_agent_workflow = fake_run  # type: ignore[assignment]
    coord._extract_response = fake_extract  # type: ignore[assignment]

    router = object()
    resp = coord.process_query("hi", settings_override={"router_engine": router})
    assert getattr(resp, "content", "") == "answer"
    assert captured["tools_data"].get("router_engine") is router
