"""Chat page staleness tests.

Covers staleness computation from manifest vs current corpus and config.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.persistence.snapshot import compute_config_hash, compute_corpus_hash
from src.persistence.snapshot_utils import compute_staleness
from src.persistence.upload_journal import quarantine_upload

_OLD_COLLECTIONS = {"text": "text__old", "image": "image__old"}
_NEW_COLLECTIONS = {"text": "text__new", "image": "image__new"}


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
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    a = uploads / "a.txt"
    a.write_text("x")
    corpus = [a]
    cfg = {
        "router": "x",
        "hybrid": False,
        "graph_enabled": False,
        "chunk_size": 128,
        "chunk_overlap": 16,
    }
    chash = compute_corpus_hash(corpus, base_dir=uploads)
    cfg_hash = compute_config_hash(cfg)
    manifest = {"corpus_hash": chash, "config_hash": cfg_hash}
    assert (
        compute_staleness(
            manifest,
            corpus,
            cfg,
            settings_obj=SimpleNamespace(data_dir=tmp_path),
        )
        is False
    )


def test_compute_staleness_keeps_current_during_source_commit(tmp_path: Path) -> None:
    """A journal bridges the narrow source-to-CURRENT activation window."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    source = uploads / "a.txt"
    source.write_text("old", encoding="utf-8")
    cfg = {"router": "x"}
    manifest = {
        "corpus_hash": compute_corpus_hash([source], base_dir=uploads),
        "config_hash": compute_config_hash(cfg),
        "collections": _OLD_COLLECTIONS,
    }
    quarantine_upload(
        data_dir=tmp_path,
        source_path=source,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
    )

    assert (
        compute_staleness(
            manifest,
            [],
            cfg,
            settings_obj=SimpleNamespace(data_dir=tmp_path),
        )
        is False
    )


def test_compute_staleness_rejects_unbound_source_journal(tmp_path: Path) -> None:
    """A journal-shaped marker cannot suppress a real corpus mismatch."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    source = uploads / "a.txt"
    source.write_text("old", encoding="utf-8")
    cfg = {"router": "x"}
    manifest = {
        "corpus_hash": compute_corpus_hash([source], base_dir=uploads),
        "config_hash": compute_config_hash(cfg),
        "collections": _OLD_COLLECTIONS,
    }
    source.write_text("new", encoding="utf-8")
    transaction = tmp_path / ".upload-transactions" / "build"
    transaction.mkdir(parents=True)
    (transaction / "transaction.json").write_text("{}", encoding="utf-8")

    assert compute_staleness(
        manifest,
        [source],
        cfg,
        settings_obj=SimpleNamespace(data_dir=tmp_path),
    )
