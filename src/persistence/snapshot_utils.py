"""Snapshot utility helpers for staleness detection and config/corpus collection.

These functions are used by Streamlit pages and other modules to compute
snapshot staleness based on the current corpus and configuration. They are
production utilities (not test-only) and should remain lightweight.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.config.settings import settings as default_settings
from src.persistence.snapshot import compute_config_hash, compute_corpus_hash


def collect_corpus_paths(base: Path) -> list[Path]:
    """Collect files under uploads directory for hashing.

    Args:
        base: Base directory to search (typically settings.data_dir / "uploads").

    Returns:
        List of file paths under ``base`` (recursive), or an empty list if the
        directory does not exist.
    """
    if not base.exists():
        return []
    return [p for p in base.glob("**/*") if p.is_file()]


def current_config_dict(settings_obj: Any | None = None) -> dict[str, Any]:
    """Build current retrieval/config dict used in config_hash.

    Args:
        settings_obj: Settings object; defaults to global settings.

    Returns:
        Dict containing configuration parameters that affect snapshot staleness.
    """
    s = settings_obj or default_settings
    return {
        "router": s.retrieval.router,
        "hybrid": s.retrieval.enable_server_hybrid,
        "graph_enabled": getattr(s, "enable_graphrag", True),
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
    """Return True when corpus/config hashes differ from manifest values.

    Prefers base_dir-normalized hashing (POSIX relpaths) for robustness;
    falls back to absolute-path hashing for older manifests.
    """
    s = settings_obj or default_settings
    uploads_dir = s.data_dir / "uploads"
    chash_norm = compute_corpus_hash(list(corpus_paths), base_dir=uploads_dir)
    cfg_hash = compute_config_hash(cfg)
    if manifest.get("config_hash") != cfg_hash:
        return True
    if manifest.get("corpus_hash") == chash_norm:
        return False
    chash_abs = compute_corpus_hash(list(corpus_paths))
    return manifest.get("corpus_hash") != chash_abs


__all__ = [
    "collect_corpus_paths",
    "compute_staleness",
    "current_config_dict",
]
