"""Shared helpers for evaluation CLI integration tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def run_beir_cli(tmp_path: Path, *, sample_count: int | None = None) -> None:
    """Run the BEIR CLI with mocked dependencies."""
    pytest.importorskip("tools.eval.run_beir")
    with (
        patch("tools.eval.run_beir.GenericDataLoader") as gdl,
        patch("tools.eval.run_beir.EvaluateRetrieval") as er,
        patch("tools.eval.run_beir.ServerHybridRetriever.retrieve") as retr,
        patch("tools.eval.run_beir.QdrantClient") as _qdrant,
        patch("src.retrieval.hybrid.QdrantClient", create=True) as _hybrid_qdrant,
        patch("src.retrieval.hybrid.check_hybrid_collection") as check_hybrid,
    ):
        check_hybrid.return_value.compatible = True
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

        from tools.eval.run_beir import main as beir_main

        data_dir = tmp_path / "scifact"
        data_dir.mkdir()
        argv = [
            "x",
            "--data_dir",
            str(data_dir),
            "--k",
            "10",
            "--results_dir",
            str(tmp_path),
        ]
        if sample_count is not None:
            argv.extend(["--sample_count", str(sample_count)])
        with patch.object(sys, "argv", argv):
            beir_main()
