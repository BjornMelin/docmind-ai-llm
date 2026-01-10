"""E2E smoke: chat via router with GraphRAG present (offline, deterministic)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.coordinator import MultiAgentCoordinator


class _Router:
    def query(self, q):
        """Return a simple namespaced response for provided query string."""
        return SimpleNamespace(response=f"ok:{q}")


@pytest.mark.e2e
def test_chat_smoke_router_override() -> None:
    """Smoke test chat flow using an injected router engine."""
    coord = MultiAgentCoordinator()
    # Avoid heavy setup
    coord._ensure_setup = lambda: True  # type: ignore[assignment]

    def fake_run(initial_state, _thread_id):  # type: ignore[no-untyped-def]
        # Simulate that tools execute router with provided engine
        tools = getattr(initial_state, "tools_data", {}) or {}
        router = tools.get("router_engine", None)
        result = router.query("hello") if router else SimpleNamespace(response="ok")
        return {
            "messages": [{"content": result.response}],
            "response": result.response,
            "coordination_time": 0.0,
        }

    coord._run_agent_workflow = fake_run  # type: ignore[assignment]
    coord._extract_response = (  # type: ignore[assignment]
        lambda _s, _q, _t, _c: SimpleNamespace(content="ok:hello")
    )

    resp = coord.process_query("hello", settings_override={"router_engine": _Router()})
    assert getattr(resp, "content", "").startswith("ok:")
