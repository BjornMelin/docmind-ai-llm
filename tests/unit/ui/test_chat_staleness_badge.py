"""Unit tests for staleness computation helper in Chat page."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_page_module(filename: str):
    base = Path(__file__).resolve().parents[3]  # project root
    path = base / "src" / "pages" / filename
    spec = importlib.util.spec_from_file_location(f"page_{filename}", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    return mod


chat_page = _load_page_module("01_chat.py")


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
