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
    # Determinism first
    set_determinism()

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
    args = ap.parse_args()
    if args.sample_count < 0:
        raise ValueError("--sample_count must be >= 0")

    df = pd.read_csv(args.dataset_csv)
    if args.sample_count > 0:
        df = df.head(args.sample_count)

    # Build answers depending on mode: offline skips online calls entirely
    if args.ragas_mode == "offline":
        answers: list[str] = [""] * len(df)
    else:
        coord = MultiAgentCoordinator()
        answers = []
        for _, row in df.iterrows():
            q = row["question"]
            resp = coord.process_query(q)
            answers.append(getattr(resp, "content", ""))

    # If dataset contains a 'contexts' column, preserve it; else empty lists
    if "contexts" in df.columns:
        ctxs = [c if isinstance(c, list) else [] for c in df["contexts"]]
    else:
        ctxs = [[] for _ in range(len(df))]

    data = pd.DataFrame(
        {
            "question": df["question"],
            "answer": answers,
            "contexts": ctxs,
            "ground_truth": df["ground_truth"],
        }
    )

    # Import ragas on-demand if not already imported (and not monkeypatched).
    global evaluate, faithfulness, answer_relevancy, context_recall, context_precision
    if evaluate is None:  # pragma: no cover - executed only without ragas in env
        try:
            from ragas import evaluate as _evaluate  # type: ignore
            from ragas.metrics import (  # type: ignore
                answer_relevancy as _ans,
            )
            from ragas.metrics import (
                context_precision as _cp,
            )
            from ragas.metrics import (
                context_recall as _cr,
            )
            from ragas.metrics import (
                faithfulness as _fh,
            )

            evaluate = _evaluate  # type: ignore[assignment]
            answer_relevancy = _ans  # type: ignore[assignment]
            context_precision = _cp  # type: ignore[assignment]
            context_recall = _cr  # type: ignore[assignment]
            faithfulness = _fh  # type: ignore[assignment]
        except Exception as exc:  # pragma: no cover - defensive
            raise ImportError(
                "ragas is required for evaluation; install optional eval extras"
            ) from exc

    # Use ragas_mode to toggle minimal runtime behavior
    show_progress = args.ragas_mode == "online_smoke"
    batch_size = 8 if args.ragas_mode == "online_smoke" else None
    result = evaluate(
        data,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        show_progress=show_progress,
        batch_size=batch_size,
    )

    # The mock in tests returns mapping of Series; handle that shape.
    # Newer ragas may return EvaluationResult with dict-like access to arrays.
    def _as_float(x: Any) -> float:
        try:
            # Pandas Series.mean or list/ndarray mean fallback
            m = getattr(x, "mean", None)
            return float(m()) if callable(m) else float(x)
        except Exception:  # pragma: no cover - defensive fallback
            return float("nan")

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
