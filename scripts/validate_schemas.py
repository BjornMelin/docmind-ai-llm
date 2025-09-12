"""Validate leaderboard CSV files against JSON Schemas and dynamic rules.

Usage:
    uv run python scripts/validate_schemas.py [--paths path1 path2 ...]

If no paths are provided, the script discovers files named 'leaderboard.csv'
recursively under the current working directory and validates them.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent


def _load_schema(name: str) -> dict:
    p = ROOT / "schemas" / name
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


BEIR_SCHEMA = Draft202012Validator(_load_schema("leaderboard_beir.schema.json"))
RAGAS_SCHEMA = Draft202012Validator(_load_schema("leaderboard_ragas.schema.json"))


def guess_leaderboard_type(header: list[str]) -> str:
    """Return leaderboard type based on header fields ("beir" or "ragas")."""
    if any(h.startswith("ndcg@") for h in header):
        return "beir"
    if {
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    }.issubset(set(header)):
        return "ragas"
    return "unknown"


def validate_file(path: Path) -> None:
    """Validate a single leaderboard CSV file against the appropriate schema."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {path}")
        header = reader.fieldnames
        kind = guess_leaderboard_type(header)
        if kind == "unknown":
            raise ValueError(
                f"Cannot determine leaderboard type for {path}; header={header}"
            )
        validator = BEIR_SCHEMA if kind == "beir" else RAGAS_SCHEMA

        # If BEIR: enforce dynamic header ↔ k consistency
        header_k: int | None = None
        if kind == "beir":
            dyn_cols = [h for h in header if re.match(r"^(ndcg|recall|mrr)@\d+$", h)]
            if not dyn_cols:
                raise ValueError(
                    "BEIR leaderboard must include at least one dynamic metric "
                    "column (ndcg@k/recall@k/mrr@k)."
                )
            ks = {int(h.split("@", 1)[1]) for h in dyn_cols}
            if len(ks) != 1:
                raise ValueError(
                    "Inconsistent dynamic metric headers; found Ks "
                    f"{sorted(ks)} in {path}"
                )
            header_k = next(iter(ks))

        for row_count, row in enumerate(reader, start=1):
            # Cast numeric fields where possible
            data: dict[str, object] = dict(row)
            if kind == "beir":
                # Required static fields
                data["k"] = int(row.get("k", 0) or 0)
                data["sample_count"] = int(row.get("sample_count", 0) or 0)
                # Validate header_k equals row k
                if header_k is not None and data["k"] != header_k:
                    raise ValueError(
                        "Row k does not match header k; "
                        f"row {row_count} has k={data['k']} while header k={header_k} "
                        f"in {path}"
                    )
                # Dynamic metric fields
                for h in header:
                    if h.startswith(("ndcg@", "recall@", "mrr@")):
                        try:
                            data[h] = float(row[h])
                        except Exception:
                            data[h] = 0.0
            else:
                # RAGAS required numeric fields
                for h in (
                    "faithfulness",
                    "answer_relevancy",
                    "context_recall",
                    "context_precision",
                ):
                    try:
                        data[h] = float(row.get(h, "nan"))
                    except Exception:
                        data[h] = 0.0
                data["sample_count"] = int(row.get("sample_count", 0) or 0)

            # Validate static fields present
            if "schema_version" not in data:
                raise ValueError(f"Missing schema_version in row {row_count} of {path}")
            if "dataset" not in data:
                raise ValueError(f"Missing dataset in row {row_count} of {path}")
            if "ts" not in data:
                raise ValueError(f"Missing ts in row {row_count} of {path}")

            errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
            if errors:
                msg = "; ".join(
                    f"{list(e.path)}: {e.message}" if e.path else e.message
                    for e in errors
                )
                raise ValueError(f"Schema validation failed for {path}: {msg}")


def discover_leaderboards() -> list[Path]:
    """Find leaderboard.csv files under the current working directory."""
    return [
        p
        for p in Path.cwd().rglob("leaderboard.csv")
        if "/.venv/" not in str(p) and "/.git/" not in str(p)
    ]


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entrypoint for validating one or more leaderboard CSV files."""
    ap = argparse.ArgumentParser(description="Validate leaderboard CSV schemas")
    ap.add_argument("--paths", nargs="*", default=[], help="Files to validate")
    args = ap.parse_args(list(argv) if argv is not None else None)

    paths: list[Path] = (
        [Path(p) for p in args.paths] if args.paths else discover_leaderboards()
    )
    ok = True
    for p in paths:
        try:
            validate_file(p)
            print(f"OK: {p}")
        except Exception as exc:  # pragma: no cover - prints to help debugging
            ok = False
            print(f"ERROR: {p}: {exc}", file=sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
