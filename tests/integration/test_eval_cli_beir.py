"""Smoke test for BEIR CLI using mocks.

Verifies that the leaderboard CSV is created without exercising heavy BEIR logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.eval_cli_helpers import run_beir_cli


@pytest.mark.integration
def test_beir_cli_smoke(tmp_path: Path) -> None:
    """Run BEIR CLI in a fully mocked environment and check output."""
    run_beir_cli(tmp_path)
    assert (tmp_path / "leaderboard.csv").exists()
