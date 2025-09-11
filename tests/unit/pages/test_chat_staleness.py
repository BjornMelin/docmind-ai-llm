"""Chat page staleness tests.

Covers staleness computation from manifest vs current corpus and config.
"""

from __future__ import annotations

from pathlib import Path

from src.persistence.snapshot import compute_config_hash, compute_corpus_hash
from src.persistence.snapshot_utils import compute_staleness


def test_compute_staleness_true(tmp_path: Path) -> None:
    """Return True when config hash differs from manifest.

    Args:
        tmp_path: Temporary directory for creating sample files.
    """
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("x")
    b.write_text("y")
    corpus = [a, b]

    cfg = {
        "router": "x",
        "hybrid": True,
        "graph_enabled": True,
        "chunk_size": 256,
        "chunk_overlap": 32,
    }
    chash = compute_corpus_hash(corpus)
    cfg_hash = compute_config_hash(cfg)
    manifest = {"corpus_hash": chash, "config_hash": cfg_hash + "diff"}
    assert compute_staleness(manifest, corpus, cfg) is True


def test_compute_staleness_false(tmp_path: Path) -> None:
    """Return False when both corpus and config hashes match manifest.

    Args:
        tmp_path: Temporary directory for creating sample files.
    """
    a = tmp_path / "a.txt"
    a.write_text("x")
    corpus = [a]
    cfg = {
        "router": "x",
        "hybrid": False,
        "graph_enabled": False,
        "chunk_size": 128,
        "chunk_overlap": 16,
    }
    chash = compute_corpus_hash(corpus)
    cfg_hash = compute_config_hash(cfg)
    manifest = {"corpus_hash": chash, "config_hash": cfg_hash}
    assert compute_staleness(manifest, corpus, cfg) is False
