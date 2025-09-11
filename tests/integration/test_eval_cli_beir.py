"""Smoke test for BEIR CLI using mocks.

Verifies that the leaderboard CSV is created without exercising heavy BEIR logic.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def test_beir_cli_smoke(tmp_path: Path) -> None:
    """Run BEIR CLI in a fully mocked environment and check output."""
    with (
        patch("tools.eval.run_beir.GenericDataLoader") as gdl,
        patch("tools.eval.run_beir.EvaluateRetrieval") as er,
        patch("tools.eval.run_beir.ServerHybridRetriever.retrieve") as retr,
    ):
        # Minimal dataset
        gdl.return_value.load.return_value = (
            {},
            {"q1": "What is AI?"},
            {"q1": {"d1": 1}},
        )

        # Retriever returns a list of SimpleNamespace-like nodes
        class _Node:
            def __init__(self, did, score):
                self.node = type("N", (), {"metadata": {"doc_id": did}})()
                self.score = score

        retr.return_value = [_Node("d1", 0.9)]
        # Evaluator returns dummy metrics
        er.return_value.evaluate.return_value = (
            {"NDCG@10": 0.5},
            None,
            {"Recall@10": 0.6},
            None,
        )
        er.return_value.evaluate_custom.return_value = {"mrr@10": 0.4}

        import sys

        from tools.eval.run_beir import main as beir_main

        data_dir = tmp_path / "scifact"
        data_dir.mkdir()
        with patch.object(
            sys,
            "argv",
            [
                "x",
                "--data_dir",
                str(data_dir),
                "--k",
                "10",
                "--results_dir",
                str(tmp_path),
            ],
        ):
            beir_main()

        assert (tmp_path / "leaderboard.csv").exists()
