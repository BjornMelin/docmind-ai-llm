"""Integration tests for the BEIR evaluation CLI header output."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def test_beir_cli_writes_dynamic_headers(tmp_path: Path) -> None:
    """Ensure leaderboard CSVs contain dynamic metric headers for BEIR runs."""
    with (
        patch("tools.eval.run_beir.GenericDataLoader") as gdl,
        patch("tools.eval.run_beir.EvaluateRetrieval") as er,
        patch("tools.eval.run_beir.ServerHybridRetriever.retrieve") as retr,
        patch("tools.eval.run_beir.QdrantClient") as _qdrant,
        patch("src.retrieval.hybrid.QdrantClient") as _hybrid_qdrant,
        patch("src.retrieval.hybrid.ensure_hybrid_collection") as _ensure_hybrid,
    ):
        # Minimal dataset
        gdl.return_value.load.return_value = (
            {},
            {"q1": "What is AI?"},
            {"q1": {"d1": 1}},
        )

        class _Node:
            def __init__(self, did: str, score: float) -> None:
                self.node = type("N", (), {"metadata": {"doc_id": did}})()
                self.score = score

        retr.return_value = [_Node("d1", 0.9)]
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
                "--sample_count",
                "1",
            ],
        ):
            beir_main()

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
