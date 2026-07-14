"""Integration coverage for coordinator timeout handling (SPEC-040)."""

from __future__ import annotations

import threading
import time
from typing import NotRequired, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.coordinator import MultiAgentCoordinator, _checkpoint_config


class _TimeoutState(TypedDict):
    """Minimal persisted graph state for timeout integration coverage."""

    total_start_time: float
    deadline_ts: float
    result: NotRequired[str]
    timed_out: NotRequired[bool]
    deadline_s: NotRequired[float]
    cancel_reason: NotRequired[str]


@pytest.mark.integration
def test_sync_node_timeout_returns_promptly_without_late_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sleeping sync node cannot publish a result after caller timeout."""
    started = threading.Event()
    finished = threading.Event()
    events: list[dict[str, object]] = []
    node_delay = threading.Event()

    def slow_node(_state: _TimeoutState) -> dict[str, str]:
        started.set()
        node_delay.wait(timeout=0.3)
        finished.set()
        return {"result": "late"}

    builder = StateGraph(_TimeoutState)
    builder.add_node("slow", slow_node)
    builder.add_edge(START, "slow")
    builder.add_edge("slow", END)
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    monkeypatch.setattr(
        "src.agents.coordinator.log_jsonl", lambda event: events.append(dict(event))
    )
    coordinator = MultiAgentCoordinator(
        max_agent_timeout=0.05,
        checkpointer=checkpointer,
    )
    coordinator.compiled_graph = graph
    initial_state: _TimeoutState = {
        "total_start_time": time.perf_counter(),
        "deadline_ts": time.monotonic() + 0.05,
    }
    config = _checkpoint_config(thread_id="timeout-thread", user_id="local")
    configurable = config["configurable"]
    assert configurable["thread_id"] != "timeout-thread"
    assert configurable["public_thread_id"] == "timeout-thread"

    try:
        started_at = time.perf_counter()
        result = coordinator._run_agent_workflow(
            dict(initial_state),
            thread_id="timeout-thread",
            user_id="local",
            checkpoint_id=None,
            runtime_context=None,
        )
        elapsed = time.perf_counter() - started_at

        assert started.is_set()
        assert elapsed < 0.2
        assert result.get("timed_out") is True, result
        assert result["deadline_s"] == 0.05
        assert result["cancel_reason"] == "deadline_exceeded"
        assert any(event.get("agent_deadline_exceeded") is True for event in events)

        immediate = graph.get_state(config)
        assert finished.wait(timeout=1.0)
        time.sleep(0.05)
        later = graph.get_state(config)

        assert "result" not in immediate.values
        assert "result" not in later.values
        assert immediate.values == later.values
    finally:
        finished.wait(timeout=1.0)
        coordinator.close()
