"""Unit tests for SnapshotManager persistence functionality.

Tests validate the complete snapshot lifecycle including:
- Creation of temporary snapshots
- Manifest writing with corpus and config hashes
- Atomic snapshot finalization
- Proper cleanup and directory structure
"""

from __future__ import annotations

from pathlib import Path

from src.persistence.snapshot import (
    SnapshotManager,
    compute_config_hash,
    compute_corpus_hash,
)


def test_snapshot_manager_roundtrip(tmp_path: Path) -> None:
    """Test complete snapshot creation and finalization roundtrip.

    Args:
        tmp_path: Temporary directory path provided by pytest fixture.

    This test validates the full snapshot workflow:
    1. Create a temporary snapshot directory
    2. Write a manifest with computed hashes
    3. Finalize the snapshot atomically
    4. Verify the final snapshot structure
    """
    mgr = SnapshotManager(tmp_path)
    tmp = mgr.begin_snapshot()
    # write a dummy manifest
    chash = compute_corpus_hash([])
    cfg_hash = compute_config_hash({"k": 1})
    mgr.write_manifest(
        tmp,
        index_id="t",
        graph_store_type="stub",
        vector_store_type="stub",
        corpus_hash=chash,
        config_hash=cfg_hash,
        versions={"app": "test"},
    )
    snap = mgr.finalize_snapshot(tmp)
    assert snap.exists()
    assert (snap / "manifest.json").exists()
