"""Unit tests for deadline propagation state seeding (SPEC-040)."""

from __future__ import annotations

import pytest

from src.agents.coordinator import MultiAgentCoordinator

pytestmark = pytest.mark.unit


def test_deadline_ts_is_always_seeded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed the mandatory absolute deadline for every coordinator run.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr("src.agents.coordinator.time.monotonic", lambda: 100.0)

    coord = MultiAgentCoordinator(max_agent_timeout=12.0)
    state = coord._build_initial_state("q", start_time=0.0)

    assert state.get("deadline_ts") == 112.0
