"""Shared CSV I/O helpers for evaluation leaderboards.

Provides a single `SCHEMA_VERSION` and a safe writer that uses the csv module
to avoid quoting issues and enforces header stability across appends.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    """Append a row to a CSV file, writing a header if needed.

    Enforces header stability across writes. Uses csv.DictWriter to ensure
    proper quoting when values contain commas.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Validate header matches existing header
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing = next(reader, None)
        if existing is not None and existing != list(row.keys()):
            raise ValueError(
                "Leaderboard schema mismatch; use a new file or bump schema_version"
            )
        # Append the row
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
    else:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)


__all__ = ["SCHEMA_VERSION", "write_csv_row"]
