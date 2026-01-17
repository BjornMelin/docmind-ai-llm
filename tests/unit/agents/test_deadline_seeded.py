"""Unit tests for deadline propagation state seeding (SPEC-040)."""

from __future__ import annotations

import pytest

from src.agents.coordinator import MultiAgentCoordinator
from src.config import settings

pytestmark = pytest.mark.unit


def test_deadline_ts_seeded_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.agents, "enable_deadline_propagation", True)
    monkeypatch.setattr("src.agents.coordinator.time.monotonic", lambda: 100.0)

    coord = MultiAgentCoordinator(max_agent_timeout=12.0, enable_fallback=False)
    state = coord._build_initial_state("q", start_time=0.0, tools_data={})

    assert state.get("deadline_ts") == 112.0


def test_deadline_ts_omitted_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.agents, "enable_deadline_propagation", False)
    monkeypatch.setattr("src.agents.coordinator.time.monotonic", lambda: 100.0)

    coord = MultiAgentCoordinator(max_agent_timeout=12.0, enable_fallback=False)
    state = coord._build_initial_state("q", start_time=0.0, tools_data={})

    assert state.get("deadline_ts") is None
