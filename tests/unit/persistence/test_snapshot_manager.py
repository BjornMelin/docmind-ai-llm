"""Unit tests for SnapshotManager persistence functionality.

Tests validate the complete snapshot lifecycle including:
- Creation of temporary snapshots
- Manifest writing with corpus and config hashes
- Atomic snapshot finalization
- Proper cleanup and directory structure
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

import src.persistence.snapshot as snapshot
from src.persistence import snapshot_writer as writer
from src.persistence.snapshot import (
    SnapshotManager,
    SnapshotPersistenceError,
    compute_config_hash,
    compute_corpus_hash,
    latest_snapshot_dir,
    load_manifest,
)
from src.persistence.snapshot_utils import current_config_dict


def _write_complete_manifest(directory: Path, **metadata: object) -> None:
    activation_config: dict[str, object] = {}
    manifest_meta: dict[str, object] = {
        "index_id": directory.name,
        "graph_store_type": "none",
        "vector_store_type": "qdrant",
        "collections": {"text": "physical-text", "image": "physical-image"},
        "corpus_hash": "0" * 64,
        "config_hash": "1" * 64,
        "versions": {},
        "graph_exports": [],
        "collection_metadata": {},
        "activation_config": activation_config,
        "activation_config_hash": compute_config_hash(activation_config),
    }
    manifest_meta.update(metadata)
    writer.write_manifest(directory, manifest_meta=manifest_meta)
    writer.mark_manifest_complete(directory)


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
        graph_store_type="none",
        vector_store_type="qdrant",
        text_collection="physical-text",
        image_collection="physical-image",
        corpus_hash=chash,
        config_hash=cfg_hash,
        versions={"app": "test"},
    )
    finalized = mgr.finalize_snapshot(tmp)
    snap = finalized.path
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
    assert payload["schema_version"] == "2.0"
    assert payload["persist_format_version"]
    assert payload["collections"] == {
        "text": "physical-text",
        "image": "physical-image",
    }

    checksum_payload = json.loads(manifest_checksum.read_text(encoding="utf-8"))
    assert checksum_payload["manifest_sha256"]
    assert checksum_payload["schema_version"] == "2.0"

    current_pointer = tmp_path / "CURRENT"
    assert current_pointer.read_text(encoding="utf-8").strip() == snap.name


def test_latest_snapshot_prefers_current_pointer(tmp_path: Path) -> None:
    """latest_snapshot_dir resolves the CURRENT pointer when present."""
    storage = tmp_path / "storage"
    storage.mkdir()
    first = storage / "20250101T000000-aaaaaaaa"
    second = storage / "20250102T000000-bbbbbbbb"
    for directory in (first, second):
        directory.mkdir()
        _write_complete_manifest(directory)
    current = storage / "CURRENT"
    current.write_text(first.name, encoding="utf-8")
    resolved = latest_snapshot_dir(storage)
    assert resolved == first


def test_snapshot_readers_fail_closed_when_current_is_incomplete(
    tmp_path: Path,
) -> None:
    """Readers never infer an activation from unreferenced directories."""
    storage = tmp_path / "storage"
    storage.mkdir()
    complete = storage / "20250101T000000-aaaaaaaa"
    incomplete = storage / "20250102T000000-bbbbbbbb"
    complete.mkdir()
    incomplete.mkdir()
    _write_complete_manifest(complete, index_id="complete")
    (incomplete / "manifest.meta.json").write_text(
        json.dumps({"complete": False, "index_id": "incomplete"}),
        encoding="utf-8",
    )
    (storage / "CURRENT").write_text(incomplete.name, encoding="utf-8")

    assert latest_snapshot_dir(storage) is None
    assert load_manifest(incomplete) is None
    assert load_manifest(base_dir=storage) is None
    assert complete.is_dir()


def test_snapshot_retention_preserves_current_and_incomplete_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GC deletes only old complete snapshots and never the CURRENT owner."""
    storage = tmp_path / "storage"
    storage.mkdir()
    names = [
        "20250101T000000-aaaaaaaa",
        "20250102T000000-bbbbbbbb",
        "20250103T000000-cccccccc",
    ]
    snapshots = [storage / name for name in names]
    for directory in snapshots:
        directory.mkdir()
        _write_complete_manifest(directory)
    incomplete = storage / "20250100T000000-incomplete"
    incomplete.mkdir()
    (incomplete / "manifest.meta.json").write_text(
        json.dumps({"complete": False}), encoding="utf-8"
    )
    (storage / "CURRENT").write_text(snapshots[0].name, encoding="utf-8")
    monkeypatch.setattr(snapshot.settings.snapshots, "retention_count", 1)
    monkeypatch.setattr(snapshot.settings.snapshots, "gc_grace_seconds", 0)

    snapshot._garbage_collect(snapshot._snapshot_paths(storage))

    assert snapshots[0].is_dir()
    assert not snapshots[1].exists()
    assert snapshots[2].is_dir()
    assert incomplete.is_dir()


def test_snapshot_retention_preserves_operator_complete_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retention owns only canonical snapshot generations."""
    storage = tmp_path / "storage"
    storage.mkdir()
    canonical = [
        storage / "20250101T000000-aaaaaaaa",
        storage / "20250102T000000-bbbbbbbb",
    ]
    operator_copy = storage / "000-operator-copy"
    for directory in (*canonical, operator_copy):
        directory.mkdir()
        _write_complete_manifest(directory)
    (storage / "CURRENT").write_text(canonical[1].name, encoding="utf-8")
    monkeypatch.setattr(snapshot.settings.snapshots, "retention_count", 1)
    monkeypatch.setattr(snapshot.settings.snapshots, "gc_grace_seconds", 0)

    snapshot._garbage_collect(snapshot._snapshot_paths(storage))

    assert operator_copy.is_dir()


def test_snapshot_retention_skips_invalid_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid CURRENT pointer fails closed instead of deleting history."""
    storage = tmp_path / "storage"
    storage.mkdir()
    snapshots = [storage / f"2025010{day}T000000-snap" for day in (1, 2)]
    for directory in snapshots:
        directory.mkdir()
        _write_complete_manifest(directory)
    (storage / "CURRENT").write_text("../escape", encoding="utf-8")
    monkeypatch.setattr(snapshot.settings.snapshots, "retention_count", 1)
    monkeypatch.setattr(snapshot.settings.snapshots, "gc_grace_seconds", 0)

    snapshot._garbage_collect(snapshot._snapshot_paths(storage))

    assert all(directory.is_dir() for directory in snapshots)


def test_config_hash_matches_current_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default compute_config_hash matches snapshot_utils current config."""
    from src.config.settings import settings as _settings

    monkeypatch.setattr(_settings.retrieval, "enable_server_hybrid", True)
    monkeypatch.setattr(_settings.graphrag_cfg, "enabled", True)
    monkeypatch.setattr(_settings.processing, "chunk_size", 128)
    monkeypatch.setattr(_settings.processing, "chunk_overlap", 16)

    cfg_dict = current_config_dict(_settings)
    hash_a = compute_config_hash(cfg_dict)
    reordered = dict(reversed(list(cfg_dict.items())))
    hash_b = compute_config_hash(reordered)
    assert hash_a == hash_b


def test_snapshot_workspace_does_not_claim_qdrant_payloads(tmp_path: Path) -> None:
    """Application snapshots leave point-in-time vectors to Qdrant backups."""
    manager = SnapshotManager(tmp_path)
    workspace = manager.begin_snapshot()
    try:
        assert not (workspace / "vector").exists()
        assert (workspace / "graph").is_dir()
    finally:
        manager.cleanup_tmp(workspace)


def test_persist_graph_storage_context_failure_records_error(tmp_path: Path) -> None:
    """Errors during graph persistence are recorded to errors.jsonl."""
    workspace = tmp_path / "_tmp-graph"
    graph_dir = workspace / "graph"
    workspace.mkdir(parents=True, exist_ok=True)

    class _FailingStorageContext:
        def persist(self, persist_dir: str) -> None:  # type: ignore[unused-argument]
            raise RuntimeError("graph-broke")

    with pytest.raises(SnapshotPersistenceError):
        snapshot.persist_graph_storage_context(_FailingStorageContext(), graph_dir)

    log_path = workspace / "errors.jsonl"
    assert log_path.exists()
    entries = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert entries
    assert any(entry.get("stage") == "persist_graph" for entry in entries)
    assert any(entry.get("error_type") == "RuntimeError" for entry in entries)
    assert any(
        str(entry.get("error", "")).startswith("[redacted:") for entry in entries
    )
    assert "graph-broke" not in log_path.read_text(encoding="utf-8")


def test_snapshot_manager_includes_graph_exports(tmp_path: Path) -> None:
    """Graph export metadata is persisted into manifest.meta.json."""
    from llama_index.core import StorageContext
    from llama_index.core.graph_stores import SimplePropertyGraphStore

    from src.utils.hashing import sha256_file

    mgr = SnapshotManager(tmp_path)
    workspace = mgr.begin_snapshot()
    mgr.persist_graph_storage_context(
        StorageContext.from_defaults(property_graph_store=SimplePropertyGraphStore()),
        workspace,
    )
    export_dir = workspace / "graph" / "graph_export-20240101T000000Z.jsonl"
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    export_dir.write_text("{}\n", encoding="utf-8")
    mgr.write_manifest(
        workspace,
        index_id="graph",
        graph_store_type="property_graph",
        vector_store_type="qdrant",
        text_collection="physical-text",
        image_collection="physical-image",
        corpus_hash="d" * 64,
        config_hash="c" * 64,
        graph_exports=[
            {
                "filename": "graph_export-20240101T000000Z.jsonl",
                "format": "jsonl",
                "created_at": "2024-01-01T00:00:00Z",
                "seed_count": 10,
                "size_bytes": export_dir.stat().st_size,
                "duration_ms": 1.0,
                "sha256": sha256_file(export_dir),
            }
        ],
        collection_metadata={
            "text": {"schema": "text-v2"},
            "image": {"schema": "image-v1"},
        },
    )
    finalized = mgr.finalize_snapshot(workspace)
    snap = finalized.path
    manifest_meta = json.loads(
        (snap / "manifest.meta.json").read_text(encoding="utf-8")
    )
    assert manifest_meta["graph_exports"][0]["filename"].startswith("graph_export-")
    assert manifest_meta["collection_metadata"] == {
        "text": {"schema": "text-v2"},
        "image": {"schema": "image-v1"},
    }
    entries = list(snapshot.load_manifest_entries(snap))
    assert any(entry["content_type"] == "application/x-ndjson" for entry in entries)


@pytest.mark.parametrize(
    "versions",
    [
        cast(dict[str, Any], {1: "invalid-key"}),
        {"app": ["invalid-value"]},
        pytest.param({"app": float("nan")}, id="nan"),
        pytest.param({"app": float("inf")}, id="positive-infinity"),
        pytest.param({"app": float("-inf")}, id="negative-infinity"),
    ],
)
def test_invalid_versions_never_replace_current(
    tmp_path: Path,
    versions: dict[str, Any],
) -> None:
    """Every version row must be projection-safe before CURRENT can move."""
    manager = SnapshotManager(tmp_path)
    prior_workspace = manager.begin_snapshot()
    manager.write_manifest(
        prior_workspace,
        index_id="prior",
        graph_store_type="none",
        vector_store_type="qdrant",
        text_collection="physical-text",
        image_collection="physical-image",
        corpus_hash="c" * 64,
        config_hash="f" * 64,
    )
    prior = manager.finalize_snapshot(prior_workspace).path
    workspace = manager.begin_snapshot()
    try:
        with pytest.raises(ValueError, match="metadata contract"):
            manager.write_manifest(
                workspace,
                index_id="invalid-versions",
                graph_store_type="none",
                vector_store_type="qdrant",
                text_collection="physical-text",
                image_collection="physical-image",
                corpus_hash="d" * 64,
                config_hash="e" * 64,
                versions=versions,
            )
    finally:
        manager.cleanup_tmp(workspace)

    assert latest_snapshot_dir(tmp_path) == prior


def test_non_finite_collection_metadata_never_writes_manifest(tmp_path: Path) -> None:
    """Strict JSON validation covers metadata outside the versions mapping."""
    manager = SnapshotManager(tmp_path)
    workspace = manager.begin_snapshot()
    try:
        with pytest.raises(ValueError, match="metadata contract"):
            manager.write_manifest(
                workspace,
                index_id="invalid-collection-metadata",
                graph_store_type="none",
                vector_store_type="qdrant",
                text_collection="physical-text",
                image_collection="physical-image",
                corpus_hash="d" * 64,
                config_hash="e" * 64,
                collection_metadata={"probe": float("nan")},
            )
        assert not (workspace / "manifest.jsonl").exists()
        assert not (workspace / "manifest.meta.json").exists()
        assert not (workspace / "manifest.checksum").exists()
    finally:
        manager.cleanup_tmp(workspace)


@pytest.mark.parametrize("format_name", [None, "x" * 33])
def test_invalid_graph_export_format_never_replaces_current(
    tmp_path: Path,
    format_name: str | None,
) -> None:
    """Missing and oversized export formats fail before CURRENT commit."""
    from llama_index.core import StorageContext
    from llama_index.core.graph_stores import SimplePropertyGraphStore

    from src.utils.hashing import sha256_file

    manager = SnapshotManager(tmp_path)
    prior_workspace = manager.begin_snapshot()
    manager.write_manifest(
        prior_workspace,
        index_id="prior",
        graph_store_type="none",
        vector_store_type="qdrant",
        text_collection="physical-text",
        image_collection="physical-image",
        corpus_hash="c" * 64,
        config_hash="f" * 64,
    )
    prior = manager.finalize_snapshot(prior_workspace).path
    workspace = manager.begin_snapshot()
    manager.persist_graph_storage_context(
        StorageContext.from_defaults(property_graph_store=SimplePropertyGraphStore()),
        workspace,
    )
    export = workspace / "graph" / "graph_export.jsonl"
    export.write_text("{}\n", encoding="utf-8")
    export_metadata: dict[str, Any] = {
        "filename": export.name,
        "size_bytes": export.stat().st_size,
        "sha256": sha256_file(export),
    }
    if format_name is not None:
        export_metadata["format"] = format_name
    try:
        with pytest.raises(ValueError, match="metadata contract"):
            manager.write_manifest(
                workspace,
                index_id="invalid-export",
                graph_store_type="property_graph",
                vector_store_type="qdrant",
                text_collection="physical-text",
                image_collection="physical-image",
                corpus_hash="d" * 64,
                config_hash="e" * 64,
                graph_exports=[export_metadata],
            )
    finally:
        manager.cleanup_tmp(workspace)

    assert latest_snapshot_dir(tmp_path) == prior


def test_snapshot_manifest_rejects_native_graph_file_as_export(
    tmp_path: Path,
) -> None:
    """Export metadata cannot alias one of the required native graph artifacts."""
    from llama_index.core import StorageContext
    from llama_index.core.graph_stores import SimplePropertyGraphStore

    from src.utils.hashing import sha256_file

    manager = SnapshotManager(tmp_path)
    workspace = manager.begin_snapshot()
    manager.persist_graph_storage_context(
        StorageContext.from_defaults(property_graph_store=SimplePropertyGraphStore()),
        workspace,
    )
    native_file = workspace / "graph" / "docstore.json"
    try:
        with pytest.raises(ValueError, match="metadata contract"):
            manager.write_manifest(
                workspace,
                index_id="graph",
                graph_store_type="property_graph",
                vector_store_type="qdrant",
                text_collection="physical-text",
                image_collection="physical-image",
                corpus_hash="d" * 64,
                config_hash="c" * 64,
                graph_exports=[
                    {
                        "filename": native_file.name,
                        "format": "jsonl",
                        "created_at": "2024-01-01T00:00:00Z",
                        "seed_count": 0,
                        "size_bytes": native_file.stat().st_size,
                        "duration_ms": 1.0,
                        "sha256": sha256_file(native_file),
                    }
                ],
            )
    finally:
        manager.cleanup_tmp(workspace)


def test_manager_initialization_does_not_recover_without_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing a manager performs no unlocked recovery writes."""
    monkeypatch.setattr(
        snapshot,
        "recover_snapshots",
        lambda *_args, **_kwargs: pytest.fail("unlocked recovery"),
    )

    SnapshotManager(tmp_path)


def test_begin_snapshot_recovers_only_after_lock_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crash cleanup observes the live writer lock and preserves its heartbeat."""
    stale_workspace = tmp_path / "_tmp-stale"
    stale_workspace.mkdir()
    original_recover = snapshot.recover_snapshots
    observed_active_lock: list[bool] = []

    def _recover(base_dir: Path | None = None) -> None:
        active = snapshot._get_active_lock()
        observed_active_lock.append(active is not None and active.path.exists())
        original_recover(base_dir)

    monkeypatch.setattr(snapshot, "recover_snapshots", _recover)
    manager = SnapshotManager(tmp_path)
    assert stale_workspace.exists()

    workspace = manager.begin_snapshot()
    try:
        assert observed_active_lock == [True]
        assert not stale_workspace.exists()
        assert (tmp_path / ".lock.meta.json").is_file()
    finally:
        manager.cleanup_tmp(workspace)


def test_finalize_crash_discards_unreferenced_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after promotion but before CURRENT leaves no inferred activation."""
    manager = SnapshotManager(tmp_path)
    workspace = manager.begin_snapshot()
    manager.write_manifest(
        workspace,
        index_id="crash",
        graph_store_type="none",
        vector_store_type="qdrant",
        text_collection="physical-text",
        image_collection="physical-image",
        corpus_hash="c" * 64,
        config_hash="f" * 64,
    )

    def _crash(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated CURRENT write crash")

    monkeypatch.setattr(snapshot, "_write_current_pointer", _crash)
    with pytest.raises(OSError, match="simulated CURRENT"):
        manager.finalize_snapshot(workspace)

    promoted = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and not path.name.startswith("_tmp-")
    ]
    assert promoted == []
    assert latest_snapshot_dir(tmp_path) is None

    snapshot.recover_snapshots(tmp_path)
    assert latest_snapshot_dir(tmp_path) is None
    assert not (tmp_path / ".activation-transaction.json").exists()


def test_finalize_verifies_promoted_manifest_before_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-rename corruption cannot cross the activation commit point."""
    manager = SnapshotManager(tmp_path)
    workspace = manager.begin_snapshot()
    manager.write_manifest(
        workspace,
        index_id="verify",
        graph_store_type="none",
        vector_store_type="qdrant",
        text_collection="physical-text",
        image_collection="physical-image",
        corpus_hash="c" * 64,
        config_hash="f" * 64,
    )
    mark_complete = snapshot.mark_manifest_complete

    def _mark_then_corrupt(destination: Path) -> None:
        mark_complete(destination)
        (destination / "manifest.meta.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(snapshot, "mark_manifest_complete", _mark_then_corrupt)

    with pytest.raises(snapshot.SnapshotPromotionError, match="verification"):
        manager.finalize_snapshot(workspace)

    assert latest_snapshot_dir(tmp_path) is None
    assert not (tmp_path / "CURRENT").exists()


def test_post_commit_observability_failure_does_not_report_rebuild_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing after CURRENT replacement can reverse a committed rebuild."""
    manager = SnapshotManager(tmp_path)
    workspace = manager.begin_snapshot()
    manager.write_manifest(
        workspace,
        index_id="commit",
        graph_store_type="none",
        vector_store_type="qdrant",
        text_collection="physical-text",
        image_collection="physical-image",
        corpus_hash="c" * 64,
        config_hash="f" * 64,
    )

    monkeypatch.setattr(
        snapshot,
        "_emit_snapshot_log",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("telemetry")),
    )

    final = manager.finalize_snapshot(workspace).path

    assert latest_snapshot_dir(tmp_path) == final


def test_finalize_retires_upload_journals_while_writer_lock_is_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-journal retirement stays inside the snapshot commit lock."""
    manager = SnapshotManager(tmp_path)
    workspace = manager.begin_snapshot()
    manager.write_manifest(
        workspace,
        index_id="locked-retirement",
        graph_store_type="none",
        vector_store_type="qdrant",
        text_collection="physical-text",
        image_collection="physical-image",
        corpus_hash="c" * 64,
        config_hash="f" * 64,
    )
    observed: list[tuple[Path, dict[str, str], bool]] = []

    def _recover_uploads(
        *, data_dir: Path, active_collections: dict[str, str] | None
    ) -> None:
        active_lock = snapshot._get_active_lock()
        observed.append(
            (
                data_dir,
                dict(active_collections or {}),
                active_lock is not None and active_lock.path.exists(),
            )
        )

    monkeypatch.setattr(snapshot, "recover_upload_quarantines", _recover_uploads)

    manager.finalize_snapshot(workspace)

    assert observed == [
        (
            tmp_path.parent,
            {"text": "physical-text", "image": "physical-image"},
            True,
        )
    ]


def test_finalization_returns_manifest_without_post_commit_snapshot_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Durable success carries the verified manifest through writer release."""
    manager = SnapshotManager(tmp_path)
    workspace = manager.begin_snapshot()
    manager.write_manifest(
        workspace,
        index_id="captured-manifest",
        graph_store_type="none",
        vector_store_type="qdrant",
        text_collection="physical-text",
        image_collection="physical-image",
        corpus_hash="c" * 64,
        config_hash="f" * 64,
    )
    monkeypatch.setattr(
        snapshot,
        "_load_complete_manifest",
        lambda _path: pytest.fail("finalized result snapshot was reopened"),
    )

    finalized = manager.finalize_snapshot(workspace)

    assert finalized.path.is_dir()
    assert finalized.manifest["index_id"] == "captured-manifest"
    assert finalized.manifest["collections"] == {
        "text": "physical-text",
        "image": "physical-image",
    }


def test_current_commit_recovery_returns_captured_manifest_without_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durability-confirmation error reuses the pre-commit verified manifest."""
    manager = SnapshotManager(tmp_path)
    workspace = manager.begin_snapshot()
    manager.write_manifest(
        workspace,
        index_id="captured-recovery",
        graph_store_type="none",
        vector_store_type="qdrant",
        text_collection="physical-text",
        image_collection="physical-image",
        corpus_hash="c" * 64,
        config_hash="f" * 64,
    )
    write_current_pointer = snapshot._write_current_pointer

    def _commit_then_fail(paths: object, version_name: str) -> None:
        write_current_pointer(paths, version_name)  # type: ignore[arg-type]
        raise OSError("simulated durability confirmation failure")

    monkeypatch.setattr(snapshot, "_write_current_pointer", _commit_then_fail)
    monkeypatch.setattr(
        snapshot,
        "_load_complete_manifest",
        lambda _path: pytest.fail("committed result snapshot was reopened"),
    )

    finalized = manager.finalize_snapshot(workspace)

    assert (tmp_path / "CURRENT").read_text(encoding="utf-8").strip() == (
        finalized.path.name
    )
    assert finalized.path.is_dir()
    assert finalized.manifest["index_id"] == "captured-recovery"
    assert (tmp_path / ".activation-transaction.json").is_file()


def test_current_directory_fsync_failure_defers_source_journal_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncertain CURRENT fsync keeps all journals for startup recovery."""
    manager = SnapshotManager(tmp_path)
    workspace = manager.begin_snapshot()
    manager.write_manifest(
        workspace,
        index_id="uncertain-current",
        graph_store_type="none",
        vector_store_type="qdrant",
        text_collection="physical-text",
        image_collection="physical-image",
        corpus_hash="c" * 64,
        config_hash="f" * 64,
    )
    original_fsync_dir = snapshot._fsync_dir
    base_fsync_count = 0
    source_recovery_calls: list[object] = []

    def _fail_current_fsync(path: Path) -> None:
        nonlocal base_fsync_count
        if path == tmp_path:
            base_fsync_count += 1
            if base_fsync_count == 3:
                raise OSError("simulated CURRENT directory fsync failure")
        original_fsync_dir(path)

    monkeypatch.setattr(snapshot, "_fsync_dir", _fail_current_fsync)
    monkeypatch.setattr(
        snapshot,
        "recover_upload_quarantines",
        lambda **kwargs: source_recovery_calls.append(kwargs),
    )

    final = manager.finalize_snapshot(workspace).path

    assert latest_snapshot_dir(tmp_path) == final
    assert (tmp_path / ".activation-transaction.json").is_file()
    assert source_recovery_calls == []

    snapshot.recover_snapshots(tmp_path)

    assert latest_snapshot_dir(tmp_path) == final
    assert not (tmp_path / ".activation-transaction.json").exists()
