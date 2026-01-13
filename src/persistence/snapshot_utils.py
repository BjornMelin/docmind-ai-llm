"""Snapshot utility helpers for staleness detection and config/corpus collection."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.settings import settings as default_settings
from src.persistence.hashing import compute_config_hash, compute_corpus_hash

__all__ = [
    "collect_corpus_paths",
    "compute_config_hash",
    "compute_staleness",
    "current_config_dict",
    "timestamped_export_path",
]


def collect_corpus_paths(base: Path) -> list[Path]:
    """Collect all files under ``base`` for inclusion in corpus hashing.

    Recursively traverses ``base`` and collects all file paths. Used to
    compute a corpus hash for staleness detection (whether documents changed).

    Args:
        base: Root directory to traverse (typically settings.data_dir /
            'uploads').

    Returns:
        List of all file paths under ``base``, or empty list if ``base``
        does not exist.
    """
    if not base.exists():
        return []
    return [p for p in base.rglob("*") if p.is_file()]


def current_config_dict(settings_obj: Any | None = None) -> dict[str, Any]:
    """Build configuration dictionary for staleness detection.

    Extracts key config parameters that affect index rebuilds. Falls back to
    `is_graphrag_enabled()` method if available, otherwise uses
    `enable_graphrag` attribute.

    Args:
        settings_obj: DocMindSettings instance, or None to use
            default_settings.

    Returns:
        Dict mapping config keys to current values (router, hybrid,
        graph_enabled, chunk_size, chunk_overlap).
    """
    s = settings_obj or default_settings
    return {
        "router": s.retrieval.router,
        "hybrid": s.retrieval.enable_server_hybrid,
        "graph_enabled": bool(
            s.is_graphrag_enabled()
            if hasattr(s, "is_graphrag_enabled")
            else getattr(s, "enable_graphrag", False)
        ),
        "chunk_size": s.processing.chunk_size,
        "chunk_overlap": s.processing.chunk_overlap,
    }


def compute_staleness(
    manifest: dict[str, Any],
    corpus_paths: Iterable[Path],
    cfg: dict[str, Any],
    *,
    settings_obj: Any | None = None,
) -> bool:
    """Return ``True`` when corpus/config hashes diverge from manifest values.

    Performs staleness detection by comparing stored hashes in the manifest against
    current hashes. Uses two corpus hash strategies:
    1. Normalized hash (with base_dir for relative paths)
    2. Absolute hash (for robustness if normalized check fails)

    Returns ``True`` (stale) immediately if config hash differs. Compares corpus hashes
    only after config is confirmed unchanged.

    Args:
        manifest: Saved manifest dict with 'config_hash' and 'corpus_hash' keys.
        corpus_paths: Iterable of file paths to hash (e.g., from collect_corpus_paths).
        cfg: Current config dict (e.g., from current_config_dict).
        settings_obj: DocMindSettings instance, or None to use default_settings.

    Returns:
        True if manifest is stale (config or corpus changed), False if up-to-date.
    """
    s = settings_obj or default_settings
    uploads_dir = s.data_dir / "uploads"
    corpus_list = list(corpus_paths)
    chash_norm = compute_corpus_hash(corpus_list, base_dir=uploads_dir)
    cfg_hash = compute_config_hash(cfg)
    if manifest.get("config_hash") != cfg_hash:
        return True
    if manifest.get("corpus_hash") == chash_norm:
        return False
    chash_abs = compute_corpus_hash(corpus_list)
    return manifest.get("corpus_hash") != chash_abs


def timestamped_export_path(out_dir: Path, extension: str) -> Path:
    """Return timestamped export path, with collision avoidance.

    Generates a filename using current UTC timestamp (YYYYMMDDTHHMMSSz
    format). If the file already exists, appends a counter to avoid
    collisions. Enforces a safety limit of 1000 attempts to prevent
    infinite loops.

    Args:
        out_dir: Directory where the export file will be created.
        extension: File extension (e.g., 'json', 'yaml') without the dot.

    Returns:
        Path with timestamped filename (e.g.,
        'graph_export-20240115T120030Z.json' or
        'graph_export-20240115T120030Z-42.json' if collision).
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    ext = extension.lstrip(".")
    if not ext:
        raise ValueError("extension must be a non-empty string")
    candidate = out_dir / f"graph_export-{ts}.{ext}"
    counter = 1
    max_attempts = 1000  # Safety limit to prevent infinite loops
    while candidate.exists() and counter < max_attempts:
        candidate = out_dir / f"graph_export-{ts}-{counter}.{ext}"
        counter += 1
    if candidate.exists():
        raise RuntimeError(
            f"Failed to generate unique export path after {counter} attempts"
        )
    return candidate
