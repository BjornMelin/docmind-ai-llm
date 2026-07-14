"""Safely find or delete orphaned DocMind Qdrant collections offline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from src.config import settings
from src.persistence.collection_cleanup import cleanup_orphan_collections
from src.utils.storage import get_client_config


def main(argv: Sequence[str] | None = None) -> int:
    """Run the quiesced orphan-collection cleanup command.

    Args:
        argv: Optional command-line arguments. Defaults to ``sys.argv``.

    Returns:
        int: Zero on success and two when cleanup fails safely.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-app-stopped",
        action="store_true",
        required=True,
        help="Acknowledge that every DocMind reader and writer is stopped.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete eligible collections; the default is a dry run.",
    )
    args = parser.parse_args(argv)

    client = QdrantClient(**get_client_config(settings))
    try:
        summary = cleanup_orphan_collections(
            client,
            delete=bool(args.delete),
            cfg=settings,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    finally:
        client.close()

    print(json.dumps(summary.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
