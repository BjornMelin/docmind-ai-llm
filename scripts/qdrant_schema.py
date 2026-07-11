"""Check or explicitly rebuild an empty DocMind Qdrant named-vector schema.

The rebuild-empty action requires all collection writers to be stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from src.config import settings
from src.utils.storage import (
    CollectionCompatibilityResult,
    check_hybrid_collection,
    get_client_config,
    rebuild_empty_hybrid_collection,
)

QUIESCENCE_NOTICE = (
    "rebuild-empty requires all collection writers to be stopped; Qdrant does "
    "not atomically lock the exact-count/delete sequence"
)


def inspect_or_rebuild(
    client: QdrantClient,
    *,
    collection: str,
    dense_dim: int,
    rebuild_empty: bool,
) -> CollectionCompatibilityResult:
    """Check compatibility and optionally rebuild only a proven-empty collection."""
    if rebuild_empty:
        return rebuild_empty_hybrid_collection(client, collection, dense_dim)
    return check_hybrid_collection(client, collection, dense_dim)


def main() -> None:
    """Run the Qdrant schema operator command."""
    parser = argparse.ArgumentParser(description=__doc__, epilog=QUIESCENCE_NOTICE)
    parser.add_argument("action", choices=["check", "rebuild-empty"])
    parser.add_argument(
        "--collection",
        default=settings.database.qdrant_collection,
    )
    parser.add_argument("--dense-dim", type=int, default=settings.embedding.dimension)
    args = parser.parse_args()

    if args.action == "rebuild-empty":
        print(QUIESCENCE_NOTICE, file=sys.stderr)

    client = QdrantClient(**get_client_config())
    try:
        result = inspect_or_rebuild(
            client,
            collection=args.collection,
            dense_dim=args.dense_dim,
            rebuild_empty=args.action == "rebuild-empty",
        )
    finally:
        client.close()
    payload: dict[str, Any] = {
        "collection": args.collection,
        "compatible": result.compatible,
        "action": result.action,
        "reason": result.reason,
        "point_count": result.point_count,
    }
    print(json.dumps(payload, sort_keys=True))
    if not result.compatible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
