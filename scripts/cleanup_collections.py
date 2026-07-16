"""Safely find or delete orphaned DocMind Qdrant collections offline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from src.config import bootstrap_settings, settings
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

    client: QdrantClient | None = None
    result: dict[str, object] | None = None
    error: Exception | None = None
    try:
        bootstrap_settings()
        client = QdrantClient(**get_client_config(settings))
        result = cleanup_orphan_collections(
            client,
            delete=bool(args.delete),
            cfg=settings,
        ).as_dict()
    except Exception as exc:
        error = exc
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                if error is None:
                    error = exc

    if error is not None:
        error_payload: dict[str, object] = {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if result is not None:
            error_payload["result"] = result
        print(
            json.dumps(error_payload, sort_keys=True),
            file=sys.stderr,
        )
        return 2

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
