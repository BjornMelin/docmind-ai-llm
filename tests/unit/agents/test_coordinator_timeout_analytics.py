"""Unit test: analytics is invoked on timeout when enabled."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.agents.coordinator import MultiAgentCoordinator


@pytest.mark.unit
def test_timeout_analytics_invoked_when_enabled(tmp_path: Path) -> None:
    coord = MultiAgentCoordinator()

    fake_settings = SimpleNamespace(
        analytics_enabled=True,
        analytics_db_path=None,
        analytics_retention_days=7,
        data_dir=tmp_path,
    )

    with (
        patch("src.agents.coordinator.settings", fake_settings),
        patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=True),
        patch.object(
            MultiAgentCoordinator,
            "_run_agent_workflow",
            return_value={"timed_out": True, "messages": []},
        ),
        patch("src.agents.coordinator.AnalyticsManager") as amock,
    ):
        amock.instance.return_value = Mock()
        coord.process_query("any")
        # Ensure log_query was called for timeout path
        assert amock.instance.return_value.log_query.called
