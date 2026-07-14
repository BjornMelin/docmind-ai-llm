"""Tests for ``src.persistence.hashing`` helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.persistence.hashing import compute_config_hash, compute_corpus_hash


def test_compute_corpus_hash_stable(tmp_path: Path) -> None:
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("alpha")
    file_b.write_text("beta")

    hash_one = compute_corpus_hash([file_a, file_b])
    hash_two = compute_corpus_hash([file_b, file_a])

    assert hash_one == hash_two


def test_compute_corpus_hash_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    with pytest.raises(ValueError, match="unavailable"):
        compute_corpus_hash([missing])


def test_compute_corpus_hash_detects_same_size_content_replacement(
    tmp_path: Path,
) -> None:
    """Hash bytes, even when size and modification time are unchanged."""
    source = tmp_path / "source.txt"
    source.write_text("alpha", encoding="utf-8")
    original_stat = source.stat()
    original_hash = compute_corpus_hash([source], base_dir=tmp_path)

    source.write_text("bravo", encoding="utf-8")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert source.stat().st_size == original_stat.st_size
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert compute_corpus_hash([source], base_dir=tmp_path) != original_hash


def test_compute_config_hash_sorted() -> None:
    config = {"chunk_size": 1024, "chunk_overlap": 100, "flags": ["a", "b"]}
    scrambled = json.loads(json.dumps(config))

    assert compute_config_hash(config) == compute_config_hash(scrambled)


def test_canonicalize_handles_paths_and_sets(tmp_path: Path) -> None:
    from src.persistence.hashing import _canonicalize

    payload = {
        "flags": {"b", "a"},
        "path": tmp_path / "sample.txt",
        "nested": (1, 2, 3),
        "ratio": 0.12345678901234,
    }
    canonical = _canonicalize(payload)

    assert canonical["flags"] == ["a", "b"]
    assert canonical["path"].endswith("sample.txt")
    assert canonical["nested"] == [1, 2, 3]
    assert canonical["ratio"] == 0.123456789012
