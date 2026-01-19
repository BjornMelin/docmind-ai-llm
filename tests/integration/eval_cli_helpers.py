"""Shared helpers for evaluation CLI integration tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
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
        patch(
            "src.retrieval.hybrid.ensure_hybrid_collection", create=True
        ) as _ensure_hybrid,
    ):
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


def run_ragas_cli(tmp_path: Path, *, sample_count: int | None = None) -> None:
    """Run the RAGAS CLI with mocked dependencies."""
    pytest.importorskip("tools.eval.run_ragas")
    if importlib.util.find_spec("ragas") is None:
        pytest.skip("requires ragas; install optional eval extras")
    csv = tmp_path / "data.csv"
    pd.DataFrame({"question": ["q1"], "ground_truth": ["gt"]}).to_csv(csv, index=False)

    with (
        patch("tools.eval.run_ragas.evaluate") as ev,
        patch(
            "src.agents.coordinator.MultiAgentCoordinator.process_query"
        ) as process_query,
    ):
        process_query.return_value = SimpleNamespace(content="answer")
        ev.return_value = {
            "faithfulness": pd.Series([1.0]),
            "answer_relevancy": pd.Series([1.0]),
            "context_recall": pd.Series([1.0]),
            "context_precision": pd.Series([1.0]),
        }

        from tools.eval.run_ragas import main as ragas_main

        argv = [
            "x",
            "--dataset_csv",
            str(csv),
            "--results_dir",
            str(tmp_path),
            "--ragas_mode",
            "offline",
        ]
        if sample_count is not None:
            argv.extend(
                [
                    "--sample_count",
                    str(sample_count),
                ]
            )
        with patch.object(sys, "argv", argv):
            ragas_main()
