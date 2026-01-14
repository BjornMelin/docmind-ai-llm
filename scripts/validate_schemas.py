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
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

ROOT = Path(__file__).resolve().parent.parent


def _load_schema(name: str) -> dict[str, Any]:
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
        header = list(reader.fieldnames)
        kind = guess_leaderboard_type(header)
        if kind == "unknown":
            raise ValueError(
                f"Cannot determine leaderboard type for {path}; header={header}"
            )
        validator = BEIR_SCHEMA if kind == "beir" else RAGAS_SCHEMA

        def get_header_k(hdr: list[str]) -> int:
            dyn = [h for h in hdr if re.match(r"^(ndcg|recall|mrr)@\d+$", h)]
            if not dyn:
                raise ValueError(
                    "BEIR leaderboard must include at least one dynamic metric column"
                )
            ks = {int(h.split("@", 1)[1]) for h in dyn}
            if len(ks) != 1:
                raise ValueError(
                    "Inconsistent dynamic metric headers; found Ks "
                    f"{sorted(ks)} in {path}"
                )
            return ks.pop()

        def normalize_beir(row: dict[str, str], header_k: int) -> dict[str, object]:
            out: dict[str, object] = dict(row)
            out["k"] = int(row.get("k", 0) or 0)
            out["sample_count"] = int(row.get("sample_count", 0) or 0)
            if out["k"] != header_k:
                raise ValueError(
                    "Row k does not match header k; "
                    f"row has k={out['k']} while header k={header_k} in {path}"
                )
            for key, val in row.items():
                if key.startswith(("ndcg@", "recall@", "mrr@")):
                    try:
                        out[key] = float(val)
                    except (TypeError, ValueError):
                        out[key] = 0.0
            return out

        def normalize_ragas(row: dict[str, str]) -> dict[str, object]:
            out: dict[str, object] = dict(row)
            for key in (
                "faithfulness",
                "answer_relevancy",
                "context_recall",
                "context_precision",
            ):
                try:
                    out[key] = float(row.get(key, "nan"))
                except (TypeError, ValueError):
                    out[key] = 0.0
            out["sample_count"] = int(row.get("sample_count", 0) or 0)
            return out

        header_k = get_header_k(header) if kind == "beir" else None

        for row_count, row in enumerate(reader, start=1):
            data = (
                normalize_beir(row, int(header_k))  # type: ignore[arg-type]
                if kind == "beir"
                else normalize_ragas(row)
            )

            # Validate static fields present
            for field in ("schema_version", "dataset", "ts"):
                if field not in data:
                    raise ValueError(f"Missing {field} in row {row_count} of {path}")

            if errors := sorted(
                validator.iter_errors(cast(Any, data)),
                key=lambda e: e.path,
            ):
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
        if ".venv" not in p.parts and ".git" not in p.parts
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
        except (
            OSError,
            ValueError,
            SchemaError,
        ) as exc:  # pragma: no cover - prints to help debugging
            ok = False
            print(f"ERROR: {p}: {exc}", file=sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
