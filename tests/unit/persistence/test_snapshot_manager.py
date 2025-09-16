"""Unit tests for SnapshotManager persistence functionality.

Tests validate the complete snapshot lifecycle including:
- Creation of temporary snapshots
- Manifest writing with corpus and config hashes
- Atomic snapshot finalization
- Proper cleanup and directory structure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.persistence.snapshot import (
    SnapshotManager,
    compute_config_hash,
    compute_corpus_hash,
    latest_snapshot_dir,
)
from src.persistence.snapshot_utils import current_config_dict


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
    stub_monitoring = ModuleType("src.utils.monitoring")
    stub_monitoring.log_performance = lambda *_, **__: None
    sys.modules.setdefault("src.utils.monitoring", stub_monitoring)
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
    manifest_json = snap / "manifest.json"
    manifest_meta = snap / "manifest.meta.json"
    manifest_jsonl = snap / "manifest.jsonl"
    manifest_checksum = snap / "manifest.checksum"

    assert manifest_json.exists()
    assert manifest_meta.exists()
    assert manifest_jsonl.exists()
    assert manifest_checksum.exists()

    payload = json.loads(manifest_meta.read_text(encoding="utf-8"))
    assert payload["corpus_hash"] == chash
    assert payload["config_hash"] == cfg_hash
    assert payload["complete"] is True
    assert payload["schema_version"]
    assert payload["persist_format_version"]

    checksum_payload = json.loads(manifest_checksum.read_text(encoding="utf-8"))
    assert checksum_payload["manifest_sha256"]
    assert checksum_payload["schema_version"] == 1

    current_pointer = tmp_path / "CURRENT"
    assert current_pointer.read_text(encoding="utf-8").strip() == snap.name


def test_latest_snapshot_prefers_current_pointer(tmp_path: Path) -> None:
    """latest_snapshot_dir resolves the CURRENT pointer when present."""
    storage = tmp_path / "storage"
    storage.mkdir()
    first = storage / "20250101T000000-aaaa"
    second = storage / "20250102T000000-bbbb"
    for directory in (first, second):
        directory.mkdir()
    current = storage / "CURRENT"
    current.write_text(first.name, encoding="utf-8")
    resolved = latest_snapshot_dir(storage)
    assert resolved == first


def test_config_hash_matches_current_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default compute_config_hash matches snapshot_utils current config."""
    from src.config.settings import settings as _settings

    monkeypatch.setattr(_settings.retrieval, "router", "auto")
    monkeypatch.setattr(_settings.retrieval, "enable_server_hybrid", True)
    monkeypatch.setattr(_settings, "enable_graphrag", True, raising=False)
    monkeypatch.setattr(_settings.processing, "chunk_size", 128)
    monkeypatch.setattr(_settings.processing, "chunk_overlap", 16)

    default_hash = compute_config_hash()
    dict_hash = compute_config_hash(current_config_dict(_settings))
    assert default_hash == dict_hash
