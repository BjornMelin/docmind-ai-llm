"""Integration tests for the BEIR evaluation CLI header output."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.eval_cli_helpers import run_beir_cli


@pytest.mark.integration
def test_beir_cli_writes_dynamic_headers(tmp_path: Path) -> None:
    """Ensure leaderboard CSVs contain dynamic metric headers for BEIR runs."""
    run_beir_cli(tmp_path, sample_count=1)
    lb = tmp_path / "leaderboard.csv"
    assert lb.exists()
    header = lb.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "schema_version" in header
    assert "dataset" in header
    assert "k" in header
    assert "sample_count" in header
    assert "ndcg@10" in header
    assert "recall@10" in header
    assert "mrr@10" in header
