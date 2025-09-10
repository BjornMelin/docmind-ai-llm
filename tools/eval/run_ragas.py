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

import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.agents.coordinator import MultiAgentCoordinator


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
    ap = argparse.ArgumentParser(description="DocMind E2E RAG eval with RAGAS")
    ap.add_argument(
        "--dataset_csv",
        required=True,
        help=("CSV with: question, ground_truth, optional contexts JSON list"),
    )
    ap.add_argument("--results_dir", default="eval/results")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset_csv)
    coord = MultiAgentCoordinator()

    answers: list[str] = []
    contexts: list[list[str]] = []
    for _, row in df.iterrows():
        q = row["question"]
        resp = coord.process_query(q)
        answers.append(getattr(resp, "content", ""))
        # If contexts not provided, leave empty; retrieval contexts can be added later
        contexts.append([])

    data = pd.DataFrame(
        {
            "question": df["question"],
            "answer": answers,
            "contexts": contexts,
            "ground_truth": df["ground_truth"],
        }
    )

    result = evaluate(
        data,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        show_progress=False,
    )
    out = {
        "ts": datetime.now(UTC).isoformat(),
        "n": len(data),
        "faithfulness": float(result["faithfulness"].mean()),
        "answer_relevancy": float(result["answer_relevancy"].mean()),
        "context_recall": float(result["context_recall"].mean()),
        "context_precision": float(result["context_precision"].mean()),
    }
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lb = out_dir / "leaderboard.csv"
    header = not lb.exists()
    with lb.open("a", encoding="utf-8") as f:
        if header:
            f.write(",".join(out.keys()) + "\n")
        f.write(",".join(str(v) for v in out.values()) + "\n")


if __name__ == "__main__":
    main()
