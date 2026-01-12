r"""RAGAS evaluation CLI for end-to-end RAG system assessment.

This module provides a command-line interface for evaluating RAG (Retrieval-Augmented
Generation) systems using RAGAS metrics. It loads a dataset of questions with ground
truth answers, queries the MultiAgentCoordinator to generate responses, and computes
comprehensive evaluation metrics including faithfulness, answer relevancy, context
recall, and context precision.

The evaluation results are automatically logged to a leaderboard CSV for tracking
performance over time and comparing different system configurations.

Typical usage example:
    python tools/eval/run_ragas.py --dataset_csv data/eval/questions.csv \
                                   --results_dir eval/results
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Optional import: allow module import even if ragas is not installed so tests can
# monkeypatch `evaluate` symbol. Real runs will import ragas at module import time
# when available, or on-demand in `main()` when not patched.
try:  # pragma: no cover - environment-dependent
    from ragas import evaluate  # type: ignore
    from ragas.metrics import (  # type: ignore
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
except Exception:  # pragma: no cover - provide placeholders for tests
    evaluate = None  # type: ignore[assignment]
    answer_relevancy = None  # type: ignore[assignment]
    context_precision = None  # type: ignore[assignment]
    context_recall = None  # type: ignore[assignment]
    faithfulness = None  # type: ignore[assignment]

from src.agents.coordinator import MultiAgentCoordinator
from src.eval.common.determinism import set_determinism
from src.eval.common.io import SCHEMA_VERSION, write_csv_row


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for RAGAS evaluation."""
    ap = argparse.ArgumentParser(description="DocMind E2E RAG eval with RAGAS")
    ap.add_argument(
        "--dataset_csv",
        required=True,
        help=("CSV with: question, ground_truth, optional contexts JSON list"),
    )
    ap.add_argument("--results_dir", default="eval/results")
    ap.add_argument(
        "--ragas_mode",
        choices=["offline", "online_smoke"],
        default="offline",
        help="Offline mocks in PR CI; limited online smoke in nightly",
    )
    ap.add_argument(
        "--sample_count",
        type=int,
        default=0,
        help="If >0, limit number of rows deterministically",
    )
    return ap


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments for RAGAS evaluation."""
    if args.sample_count < 0:
        raise ValueError("--sample_count must be >= 0")


def _load_dataset(path: str, sample_count: int) -> pd.DataFrame:
    """Load the dataset CSV and optionally limit rows."""
    df = pd.read_csv(path)
    if sample_count > 0:
        df = df.head(sample_count)
    return df


def _build_answers(df: pd.DataFrame, ragas_mode: str) -> list[str]:
    """Build answers with offline or online coordinator mode."""
    if ragas_mode == "offline":
        return [""] * len(df)
    coord = MultiAgentCoordinator()
    answers: list[str] = []
    for _, row in df.iterrows():
        q = row["question"]
        resp = coord.process_query(q)
        answers.append(getattr(resp, "content", ""))
    return answers


def _build_contexts(df: pd.DataFrame) -> list[list[Any]]:
    """Build contexts list from dataset, defaulting to empty lists."""
    if "contexts" not in df.columns:
        return [[] for _ in range(len(df))]
    return [c if isinstance(c, list) else [] for c in df["contexts"]]


def _ensure_ragas_loaded() -> None:
    """Ensure ragas modules are available, importing on demand."""
    global evaluate, faithfulness, answer_relevancy, context_recall, context_precision
    if evaluate is not None:
        return
    try:  # pragma: no cover
        from ragas import evaluate as _evaluate  # type: ignore
        from ragas.metrics import answer_relevancy as _ans  # type: ignore
        from ragas.metrics import context_precision as _cp  # type: ignore
        from ragas.metrics import context_recall as _cr  # type: ignore
        from ragas.metrics import faithfulness as _fh  # type: ignore

        evaluate = _evaluate  # type: ignore[assignment]
        answer_relevancy = _ans  # type: ignore[assignment]
        context_precision = _cp  # type: ignore[assignment]
        context_recall = _cr  # type: ignore[assignment]
        faithfulness = _fh  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover - defensive
        raise ImportError(
            "ragas is required for evaluation; install optional eval extras"
        ) from exc


def _as_float(x: Any) -> float:
    """Return a float from ragas metric results."""
    try:
        m = getattr(x, "mean", None)
        return float(m()) if callable(m) else float(x)
    except Exception:  # pragma: no cover - defensive fallback
        return float("nan")


def main() -> None:
    """Evaluate RAG system using RAGAS metrics and log results to leaderboard.

    This function loads a CSV dataset containing questions and ground truth answers,
    queries the MultiAgentCoordinator to generate responses for each question, computes
    RAGAS evaluation metrics, and appends the results to a leaderboard CSV file.

    The evaluation process includes:
    1. Loading and parsing the input CSV dataset
    2. Initializing the MultiAgentCoordinator
    3. Generating responses for each question
    4. Computing RAGAS metrics (faithfulness, relevancy, context recall/precision)
    5. Logging results to the leaderboard CSV

    Command-line arguments are parsed to configure the evaluation parameters.

    Raises:
        FileNotFoundError: If the specified dataset CSV file doesn't exist.
        ValueError: If the CSV doesn't contain required columns.
        RuntimeError: If the MultiAgentCoordinator fails to initialize.
    """
    set_determinism()
    ap = _build_parser()
    args = ap.parse_args()
    _validate_args(args)

    df = _load_dataset(args.dataset_csv, args.sample_count)
    answers = _build_answers(df, args.ragas_mode)
    ctxs = _build_contexts(df)
    data = pd.DataFrame(
        {
            "question": df["question"],
            "answer": answers,
            "contexts": ctxs,
            "ground_truth": df["ground_truth"],
        }
    )

    _ensure_ragas_loaded()

    # Use ragas_mode to toggle minimal runtime behavior
    show_progress = args.ragas_mode == "online_smoke"
    batch_size = 8 if args.ragas_mode == "online_smoke" else None
    result = evaluate(
        data,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        show_progress=show_progress,
        batch_size=batch_size,
    )

    out = {
        "schema_version": SCHEMA_VERSION,
        "ts": datetime.now(UTC).isoformat(),
        "dataset": Path(args.dataset_csv).name,
        "faithfulness": _as_float(result["faithfulness"]),
        "answer_relevancy": _as_float(result["answer_relevancy"]),
        "context_recall": _as_float(result["context_recall"]),
        "context_precision": _as_float(result["context_precision"]),
        "sample_count": len(data),
    }
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lb = out_dir / "leaderboard.csv"
    write_csv_row(lb, out)


if __name__ == "__main__":
    main()
