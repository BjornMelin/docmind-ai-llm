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
from types import ModuleType, SimpleNamespace

import pytest

import src.persistence.snapshot as snapshot
from src.persistence.snapshot import (
    SnapshotManager,
    SnapshotPersistenceError,
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
    manifest_meta = snap / "manifest.meta.json"
    manifest_jsonl = snap / "manifest.jsonl"
    manifest_checksum = snap / "manifest.checksum"

    for path_obj in (manifest_meta, manifest_jsonl, manifest_checksum):
        assert path_obj.exists()

    payload = json.loads(manifest_meta.read_text(encoding="utf-8"))
    assert payload["corpus_hash"] == chash
    assert payload["config_hash"] == cfg_hash
    assert payload["complete"] is True
    assert payload["schema_version"]
    assert payload["persist_format_version"]

    checksum_payload = json.loads(manifest_checksum.read_text(encoding="utf-8"))
    assert checksum_payload["manifest_sha256"]
    assert checksum_payload["schema_version"] == "1.0"

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

    cfg_dict = current_config_dict(_settings)
    hash_a = compute_config_hash(cfg_dict)
    reordered = dict(reversed(list(cfg_dict.items())))
    hash_b = compute_config_hash(reordered)
    assert hash_a == hash_b


def test_persist_vector_index_failure_records_error(tmp_path: Path) -> None:
    """Errors during vector persistence are recorded to errors.jsonl."""
    workspace = tmp_path / "_tmp-vec"
    vector_dir = workspace / "vector"
    workspace.mkdir(parents=True, exist_ok=True)

    class _FailingStorage:
        def persist(self, persist_dir: str) -> None:  # type: ignore[unused-argument]
            raise RuntimeError("boom")

    failing_index = SimpleNamespace(storage_context=_FailingStorage())

    with pytest.raises(SnapshotPersistenceError):
        snapshot.persist_vector_index(failing_index, vector_dir)

    log_path = workspace / "errors.jsonl"
    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8")
    assert "persist_vector" in contents
    assert "boom" in contents


def test_persist_graph_store_failure_records_error(tmp_path: Path) -> None:
    """Errors during graph persistence are recorded to errors.jsonl."""
    workspace = tmp_path / "_tmp-graph"
    graph_dir = workspace / "graph"
    workspace.mkdir(parents=True, exist_ok=True)

    class _FailingGraphStore:
        def persist(self, persist_dir: str) -> None:  # type: ignore[unused-argument]
            raise RuntimeError("graph-broke")

    with pytest.raises(SnapshotPersistenceError):
        snapshot.persist_graph_store(_FailingGraphStore(), graph_dir)

    log_path = workspace / "errors.jsonl"
    assert log_path.exists()
    data = log_path.read_text(encoding="utf-8")
    assert "persist_graph" in data
    assert "graph-broke" in data


def test_snapshot_manager_includes_graph_exports(tmp_path: Path) -> None:
    """Graph export metadata is persisted into manifest.meta.json."""
    mgr = SnapshotManager(tmp_path)
    workspace = mgr.begin_snapshot()
    export_dir = workspace / "graph" / "graph_export-20240101T000000Z.jsonl"
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    export_dir.write_text("{}\n", encoding="utf-8")
    mgr.write_manifest(
        workspace,
        index_id="graph",
        graph_store_type="simple",
        vector_store_type="qdrant",
        corpus_hash="deadbeef",
        config_hash="cafebabe",
        graph_exports=[
            {
                "filename": "graph_export-20240101T000000Z.jsonl",
                "format": "jsonl",
                "created_at": "2024-01-01T00:00:00Z",
                "seed_count": 10,
                "size_bytes": export_dir.stat().st_size,
            }
        ],
    )
    snap = mgr.finalize_snapshot(workspace)
    manifest_meta = json.loads(
        (snap / "manifest.meta.json").read_text(encoding="utf-8")
    )
    assert manifest_meta["graph_exports"][0]["filename"].startswith("graph_export-")
    entries = list(snapshot.load_manifest_entries(snap))
    assert any(entry["content_type"] == "application/x-ndjson" for entry in entries)
