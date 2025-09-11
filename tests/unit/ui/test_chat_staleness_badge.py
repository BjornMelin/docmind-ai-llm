"""Unit tests for staleness computation helper in Chat page."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.persistence.snapshot import compute_config_hash, compute_corpus_hash
from src.persistence.snapshot_utils import collect_corpus_paths, compute_staleness


@pytest.mark.unit
def test_compute_staleness_changes_with_corpus(tmp_path: Path) -> None:
    # Prepare corpus files
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    corpus = [f]
    cfg = {
        "router": "auto",
        "hybrid": True,
        "graph_enabled": True,
        "chunk_size": 512,
        "chunk_overlap": 64,
    }
    # Compute manifest with current hashes
    chash = compute_corpus_hash(corpus)
    cfg_hash = compute_config_hash(cfg)
    manifest = {"corpus_hash": chash, "config_hash": cfg_hash}
    # Not stale initially
    assert compute_staleness(manifest, corpus, cfg) is False
    # Change corpus -> stale
    f.write_text("xy", encoding="utf-8")
    assert compute_staleness(manifest, corpus, cfg) is True


@pytest.mark.unit
def test_collect_corpus_paths_handles_missing_dir(tmp_path: Path) -> None:
    paths = collect_corpus_paths(tmp_path / "missing")
    assert paths == []
