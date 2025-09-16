"""Deterministic hashing helpers for snapshot manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def compute_corpus_hash(paths: Sequence[Path]) -> str:
    """Return a stable SHA-256 hash for the provided corpus paths.

    Each path contributes its POSIX-style relative path, file size, and
    modification time in nanoseconds. The inputs are sorted to guarantee
    deterministic output regardless of traversal order.
    """
    digest = hashlib.sha256()
    for entry in sorted((Path(p).as_posix(), Path(p)) for p in paths):
        rel, path_obj = entry
        try:
            stat = path_obj.stat()
        except FileNotFoundError:
            size = -1
            mtime_ns = -1
        else:
            size = stat.st_size
            mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
        digest.update(rel.encode("utf-8"))
        digest.update(str(size).encode("utf-8"))
        digest.update(str(mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def compute_config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for ingestion configuration mappings."""
    serialized = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = ["compute_config_hash", "compute_corpus_hash"]
