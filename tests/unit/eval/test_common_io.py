"""Unit tests for ``src.eval.common.io`` helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.eval.common.io import write_csv_row


@pytest.mark.unit
def test_write_csv_row_creates_header(tmp_path: Path) -> None:
    target = tmp_path / "report.csv"
    write_csv_row(target, {"a": "x", "b": "y"})

    with target.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        assert next(reader) == ["a", "b"]
        assert next(reader) == ["x", "y"]


@pytest.mark.unit
def test_write_csv_row_enforces_header(tmp_path: Path) -> None:
    target = tmp_path / "report.csv"
    write_csv_row(target, {"a": "x", "b": "y"})

    with pytest.raises(ValueError, match="Leaderboard schema mismatch"):
        write_csv_row(target, {"b": "y", "a": "x", "c": "z"})


@pytest.mark.unit
def test_write_csv_row_appends(tmp_path: Path) -> None:
    target = tmp_path / "report.csv"
    write_csv_row(target, {"name": "row1", "value": "1,2"})
    write_csv_row(target, {"name": "row2", "value": "3,4"})

    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert lines[1] == 'row1,"1,2"'
    assert lines[2] == 'row2,"3,4"'
