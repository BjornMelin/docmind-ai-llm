"""Deterministic hashing helpers for snapshot manifests and configuration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["compute_config_hash", "compute_corpus_hash"]


def compute_corpus_hash(paths: Sequence[Path], *, base_dir: Path | None = None) -> str:
    """Return a stable SHA-256 hash for the provided corpus paths.

    Args:
        paths: Files that make up the persisted corpus snapshot.
        base_dir: Optional base directory used to normalise relative paths.

    Returns:
        str: Hex-encoded SHA-256 digest representing the corpus.
    """
    digest = hashlib.sha256()
    normalised: list[tuple[str, int, int]] = []
    resolved_base = Path(base_dir).resolve() if base_dir is not None else None
    for candidate in paths:
        path_obj = Path(candidate).resolve()
        rel = path_obj.as_posix()
        if resolved_base is not None:
            try:
                rel = path_obj.relative_to(resolved_base).as_posix()
            except ValueError:
                rel = path_obj.as_posix()
        try:
            stat = path_obj.stat()
        except FileNotFoundError:
            size = -1
            mtime_ns = -1
        else:
            size = stat.st_size
            mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
        normalised.append((rel, size, mtime_ns))
    for rel, size, mtime_ns in sorted(normalised):
        digest.update(rel.encode("utf-8"))
        digest.update(str(size).encode("utf-8"))
        digest.update(str(mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def compute_config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for ingestion configuration mappings.

    Args:
        config: Serialization-friendly mapping representing pipeline config.

    Returns:
        str: Hex-encoded SHA-256 digest for the configuration payload.
    """
    canonical = _canonicalize(config)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonicalize(value: Any) -> Any:
    """Recursively canonicalise arbitrary values for hashing."""
    if isinstance(value, Mapping):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_canonicalize(item) for item in value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        return float(f"{value:.12g}")
    return value
