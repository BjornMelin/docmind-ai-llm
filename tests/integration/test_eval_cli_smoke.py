"""Smoke tests for evaluation CLIs using mocks to avoid heavy dependencies."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd


def test_ragas_cli_smoke(tmp_path: Path) -> None:
    """Ensure RAGAS CLI writes a leaderboard row with mocked evaluate/coordinator."""
    csv = tmp_path / "data.csv"
    pd.DataFrame({"question": ["q1"], "ground_truth": ["gt"]}).to_csv(csv, index=False)

    with (
        patch("tools.eval.run_ragas.evaluate") as ev,
        patch(
            "src.agents.coordinator.MultiAgentCoordinator.process_query"
        ) as process_query,
    ):

        class _Resp:
            content = "answer"

        process_query.return_value = _Resp()
        ev.return_value = {
            "faithfulness": pd.Series([1.0]),
            "answer_relevancy": pd.Series([1.0]),
            "context_recall": pd.Series([1.0]),
            "context_precision": pd.Series([1.0]),
        }
        import sys

        from tools.eval.run_ragas import main as ragas_main

        with patch.object(
            sys,
            "argv",
            ["x", "--dataset_csv", str(csv), "--results_dir", str(tmp_path)],
        ):
            ragas_main()
        assert (tmp_path / "leaderboard.csv").exists()
