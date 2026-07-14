"""Unit test for settings_override runtime-context ownership.

This test patches internal methods of MultiAgentCoordinator to avoid heavy
dependencies and ensures that router engines never enter persisted state.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.agents.coordinator import MultiAgentCoordinator


def test_settings_override_router_engine_passthrough() -> None:
    """Router engine passed via settings_override should reach runtime context."""
    coord = MultiAgentCoordinator()

    # Patch setup to skip heavy initialization
    coord._ensure_setup = lambda: True  # type: ignore[assignment]

    captured = {}

    def fake_run(initial_state, **kwargs):  # type: ignore[no-untyped-def]
        captured["initial_state"] = dict(initial_state)
        captured["runtime_context"] = kwargs.get("runtime_context")
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
    assert "tools_data" not in captured["initial_state"]
    assert captured["runtime_context"].get("router_engine") is router
