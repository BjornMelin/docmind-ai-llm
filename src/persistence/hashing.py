"""Deterministic hashing helpers for snapshot manifests and configuration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "compute_config_hash",
    "compute_corpus_hash",
    "compute_corpus_hash_entries",
]


def compute_corpus_hash(paths: Sequence[Path], *, base_dir: Path | None = None) -> str:
    """Return a stable SHA-256 hash for corpus paths and file contents.

    Args:
        paths: Files that make up the persisted corpus snapshot.
        base_dir: Optional base directory used to normalise relative paths.

    Returns:
        str: Hex-encoded SHA-256 digest representing the corpus.
    """
    entries: list[tuple[str, Path]] = []
    resolved_base = Path(base_dir).resolve() if base_dir is not None else None
    for candidate in paths:
        candidate_path = Path(candidate)
        if candidate_path.is_symlink():
            raise ValueError("Corpus paths must not be symlinks")
        try:
            path_obj = candidate_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("Corpus path is unavailable") from exc
        if not path_obj.is_file():
            raise ValueError("Corpus path must be a regular file")
        rel = path_obj.as_posix()
        if resolved_base is not None:
            try:
                rel = path_obj.relative_to(resolved_base).as_posix()
            except ValueError as exc:
                raise ValueError("Corpus path escapes its canonical root") from exc
        entries.append((rel, path_obj))
    return compute_corpus_hash_entries(entries)


def compute_corpus_hash_entries(entries: Sequence[tuple[str, Path]]) -> str:
    """Hash content under explicit canonical logical paths.

    This supports prospective activation: bytes may remain in a staging path while
    their durable upload-relative destination participates in the corpus identity.
    """
    digest = hashlib.sha256()
    normalised: list[tuple[str, str | None]] = []
    logical_paths: set[str] = set()
    for logical_path, source_path in entries:
        if not logical_path or logical_path in logical_paths:
            raise ValueError("Corpus logical paths must be unique and non-empty")
        logical_paths.add(logical_path)
        candidate = Path(source_path)
        if candidate.is_symlink():
            raise ValueError("Corpus sources must not be symlinks")
        try:
            path_obj = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("Corpus source is unavailable") from exc
        if not path_obj.is_file():
            raise ValueError("Corpus source must be a regular file")
        with path_obj.open("rb") as source:
            content_hash = hashlib.file_digest(source, "sha256").hexdigest()
        normalised.append((logical_path, content_hash))
    for rel, content_hash in sorted(normalised):
        digest.update(
            json.dumps(
                [rel, content_hash],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
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
    """Recursively canonicalise arbitrary values for deterministic hashing.

    Converts complex types to canonical JSON-serializable forms:
    - Mapping: sorted by keys (nested canonicalization)
    - list/tuple: canonicalized elements
    - set/frozenset: sorted canonical elements
    - Path: POSIX path string for OS-independent representation
    - float: normalized to 12 significant digits to avoid floating-point noise

    Args:
        value: Any Python value (primitive, collection, or custom type).

    Returns:
        Canonical form suitable for JSON serialization and stable hashing.

    Example:
        >>> _canonicalize({'b': 1, 'a': 2})
        {'a': 2, 'b': 1}  # Keys sorted
        >>> _canonicalize(3.141592653589793)
        3.14159265359  # Limited to 12 significant digits
    """
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
