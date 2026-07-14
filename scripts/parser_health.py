"""Report DocMind parser dependency and model availability as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import settings
from src.processing.parsing.health import parser_health


def main() -> None:
    """Print parser health diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero unless PDF dependencies and Docling models are available.",
    )
    args = parser.parse_args()
    health = parser_health(settings)
    print(json.dumps(health, indent=2, sort_keys=True))
    if args.check and not health["pdf_dependencies_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
