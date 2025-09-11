"""Unit tests for staleness computation helper in Chat page."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

chat_page = importlib.import_module("src.pages.01_chat")


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
    chash = chat_page.compute_corpus_hash(corpus)
    cfg_hash = chat_page.compute_config_hash(cfg)
    manifest = {"corpus_hash": chash, "config_hash": cfg_hash}
    # Not stale initially
    assert chat_page.compute_staleness(manifest, corpus, cfg) is False
    # Change corpus -> stale
    f.write_text("xy", encoding="utf-8")
    assert chat_page.compute_staleness(manifest, corpus, cfg) is True


@pytest.mark.unit
def test_collect_corpus_paths_handles_missing_dir(tmp_path: Path) -> None:
    paths = chat_page._collect_corpus_paths(tmp_path / "missing")
    assert paths == []
