"""Unit tests for compute_corpus_hash relpath normalization and ordering.

Ensures that hash is stable across file orderings when base_dir is specified.
"""

from __future__ import annotations

from pathlib import Path

from src.persistence.snapshot import compute_corpus_hash


def test_corpus_hash_stable_across_orders(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    a = uploads / "a.txt"
    b = uploads / "sub" / "b.txt"
    b.parent.mkdir(parents=True, exist_ok=True)
    a.write_text("A", encoding="utf-8")
    b.write_text("B", encoding="utf-8")

    files = [a, b]
    h1 = compute_corpus_hash(files, base_dir=uploads)
    h2 = compute_corpus_hash(list(reversed(files)), base_dir=uploads)
    assert h1 == h2
