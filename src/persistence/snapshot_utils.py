"""Snapshot utility helpers for staleness detection and config/corpus collection."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.config.settings import settings as default_settings
from src.persistence.hashing import compute_config_hash, compute_corpus_hash

__all__ = [
    "collect_corpus_paths",
    "compute_config_hash",
    "compute_staleness",
    "current_config_dict",
]


def collect_corpus_paths(base: Path) -> list[Path]:
    """Collect files under ``base`` for inclusion in corpus hashing."""
    if not base.exists():
        return []
    return [p for p in base.rglob("*") if p.is_file()]


def current_config_dict(settings_obj: Any | None = None) -> dict[str, Any]:
    """Build the configuration dictionary used for staleness detection."""
    s = settings_obj or default_settings
    return {
        "router": s.retrieval.router,
        "hybrid": s.retrieval.enable_server_hybrid,
        "graph_enabled": bool(getattr(s, "enable_graphrag", False)),
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
    """Return ``True`` when corpus/config hashes diverge from manifest values."""
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
