"""Smoke tests for evaluation CLIs using mocks to avoid heavy dependencies."""

from __future__ import annotations

from pathlib import Path

from tests.integration.eval_cli_helpers import run_ragas_cli


def test_ragas_cli_smoke(tmp_path: Path) -> None:
    """Ensure RAGAS CLI writes a leaderboard row with mocked evaluate/coordinator."""
    run_ragas_cli(tmp_path)
    assert (tmp_path / "leaderboard.csv").exists()
