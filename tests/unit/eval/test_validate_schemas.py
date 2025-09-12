from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.validate_schemas import validate_file


@pytest.mark.unit
def test_validate_file_beir_header_k_ok(tmp_path: Path) -> None:
    lb = tmp_path / "leaderboard.csv"
    header = [
        "schema_version",
        "ts",
        "dataset",
        "k",
        "ndcg@10",
        "recall@10",
        "mrr@10",
        "sample_count",
    ]
    rows = [
        {
            "schema_version": "1.0",
            "ts": "2025-09-12T00:00:00Z",
            "dataset": "toy",
            "k": "10",
            "ndcg@10": "0.5",
            "recall@10": "0.6",
            "mrr@10": "0.4",
            "sample_count": "1",
        }
    ]
    with lb.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    # Should not raise
    validate_file(lb)


@pytest.mark.unit
def test_validate_file_beir_header_k_mismatch_raises(tmp_path: Path) -> None:
    lb = tmp_path / "leaderboard.csv"
    header = [
        "schema_version",
        "ts",
        "dataset",
        "k",
        "ndcg@10",
        "recall@10",
        "mrr@10",
        "sample_count",
    ]
    rows = [
        {
            "schema_version": "1.0",
            "ts": "2025-09-12T00:00:00Z",
            "dataset": "toy",
            "k": "5",  # mismatch vs header @10
            "ndcg@10": "0.5",
            "recall@10": "0.6",
            "mrr@10": "0.4",
            "sample_count": "1",
        }
    ]
    with lb.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="does not match header k"):
        validate_file(lb)
