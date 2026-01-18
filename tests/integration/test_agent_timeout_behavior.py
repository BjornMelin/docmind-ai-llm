"""Integration coverage for coordinator timeout handling (SPEC-040)."""

from __future__ import annotations

import pytest

from src.agents.coordinator import MultiAgentCoordinator
from src.config import settings


@pytest.mark.integration
def test_agent_deadline_exceeded_marks_state_and_emits_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify deadline propagation sets timeout state and emits an event.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr(settings.agents, "enable_deadline_propagation", True)

    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.agents.coordinator.log_jsonl", lambda ev: events.append(dict(ev))
    )

    # Deterministic clock progression: seed at t=100, then exceed at t=111.
    monotonic_values = iter([100.0, 111.0])
    monkeypatch.setattr(
        "src.agents.coordinator.time.monotonic", lambda: next(monotonic_values)
    )
    monkeypatch.setattr("src.agents.coordinator.time.perf_counter", lambda: 0.02)

    class _StubGraph:
        def stream(self, *_args, **_kwargs):
            yield {"messages": []}

    coord = MultiAgentCoordinator(max_agent_timeout=10.0, enable_fallback=False)
    coord.compiled_graph = _StubGraph()

    initial_state = coord._build_initial_state("q", start_time=0.0, tools_data={})
    result = coord._run_agent_workflow(
        initial_state,
        thread_id="t",
        user_id="u",
        checkpoint_id=None,
        runtime_context=None,
    )

    assert result.get("timed_out") is True
    assert result.get("deadline_s") == 10.0
    assert result.get("cancel_reason") == "deadline_exceeded"
    assert any(ev.get("agent_deadline_exceeded") is True for ev in events)
