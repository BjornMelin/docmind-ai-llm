"""Integration tests ensuring RAGAS CLI outputs expected headers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.eval_cli_helpers import run_ragas_cli


@pytest.mark.integration
def test_ragas_cli_writes_required_headers(tmp_path: Path) -> None:
    """Ensure leaderboard CSV contains all required RAGAS metrics."""
    run_ragas_cli(tmp_path, sample_count=1)
    lb = tmp_path / "leaderboard.csv"
    assert lb.exists()
    header = lb.read_text(encoding="utf-8").splitlines()[0].split(",")
    for col in [
        "schema_version",
        "ts",
        "dataset",
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
        "sample_count",
    ]:
        assert col in header
