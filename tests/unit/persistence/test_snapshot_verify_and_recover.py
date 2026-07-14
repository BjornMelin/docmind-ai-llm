"""Targeted unit tests for snapshot verification and recovery helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

import src.persistence.snapshot as snap
from src.persistence import snapshot_writer as writer
from src.persistence.hashing import compute_config_hash

pytestmark = pytest.mark.unit


def _manifest_metadata(index_id: str) -> dict[str, object]:
    activation_config: dict[str, object] = {}
    return {
        "index_id": index_id,
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


def _write_complete_manifest(snapshot_dir: Path) -> None:
    writer.write_manifest(
        snapshot_dir,
        manifest_meta=_manifest_metadata(snapshot_dir.name),
    )
    writer.mark_manifest_complete(snapshot_dir)


def test_verify_snapshot_roundtrip_and_mismatch(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    payload = snapshot_dir / "payload.txt"
    payload.write_text("hello", encoding="utf-8")

    writer.write_manifest(snapshot_dir, manifest_meta=_manifest_metadata("x"))
    assert snap.verify_snapshot(snapshot_dir) is False
    writer.mark_manifest_complete(snapshot_dir)
    assert snap.verify_snapshot(snapshot_dir) is True

    payload.write_text("tampered", encoding="utf-8")
    assert snap.verify_snapshot(snapshot_dir) is False


def test_manifest_emission_follows_payload_and_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest never advertises payload bytes before durable tree flushes."""
    snapshot_dir = tmp_path / "snap"
    nested = snapshot_dir / "graph"
    nested.mkdir(parents=True)
    payload = nested / "property_graph_store.json"
    payload.write_text("{}", encoding="utf-8")
    events: list[str] = []
    original_fsync_file = writer._fsync_file
    original_fsync_dir = writer._fsync_dir
    original_write_text_atomic = writer._write_text_atomic

    def _record_file(path: Path) -> None:
        events.append(f"file:{path.relative_to(snapshot_dir)}")
        original_fsync_file(path)

    def _record_dir(path: Path) -> None:
        events.append(f"dir:{path.relative_to(snapshot_dir)}")
        original_fsync_dir(path)

    def _record_manifest(path: Path, data: str) -> None:
        events.append(f"manifest:{path.name}")
        original_write_text_atomic(path, data)

    monkeypatch.setattr(writer, "_fsync_file", _record_file)
    monkeypatch.setattr(writer, "_fsync_dir", _record_dir)
    monkeypatch.setattr(writer, "_write_text_atomic", _record_manifest)

    writer.write_manifest(snapshot_dir, manifest_meta=_manifest_metadata("x"))

    first_manifest = events.index("manifest:manifest.jsonl")
    assert events.index("file:graph/property_graph_store.json") < first_manifest
    assert events.index("dir:graph") < first_manifest
    assert events.index("dir:.") < first_manifest


def test_verify_snapshot_rejects_malformed_manifest_suffix(tmp_path: Path) -> None:
    """Malformed rows cannot be ignored while retaining an old checksum."""
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    (snapshot_dir / "payload.txt").write_text("hello", encoding="utf-8")
    writer.write_manifest(snapshot_dir, manifest_meta=_manifest_metadata("x"))
    writer.mark_manifest_complete(snapshot_dir)

    with (snapshot_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{malformed}\n")

    assert snap.verify_snapshot(snapshot_dir) is False


def test_verify_snapshot_rejects_unmanifested_payload(tmp_path: Path) -> None:
    """A snapshot is a closed world of exactly its manifest payload paths."""
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    writer.write_manifest(snapshot_dir, manifest_meta=_manifest_metadata("x"))
    writer.mark_manifest_complete(snapshot_dir)
    graph_dir = snapshot_dir / "graph"
    graph_dir.mkdir()
    (graph_dir / "stale.json").write_text("{}", encoding="utf-8")

    assert snap.verify_snapshot(snapshot_dir) is False


def test_property_graph_loader_requires_manifest_graph_type(tmp_path: Path) -> None:
    """Graph artifacts cannot activate when metadata explicitly declares none."""
    snapshot_dir = tmp_path / "snap"
    graph_dir = snapshot_dir / "graph"
    graph_dir.mkdir(parents=True)
    (graph_dir / "store.json").write_text("{}", encoding="utf-8")
    writer.write_manifest(
        snapshot_dir,
        manifest_meta=_manifest_metadata("x"),
    )
    writer.mark_manifest_complete(snapshot_dir)

    assert snap.load_property_graph_index(snapshot_dir) is None


def test_recover_snapshots_keeps_current_when_valid(tmp_path: Path) -> None:
    base = tmp_path / "storage"
    base.mkdir()
    older = base / "20250101T000000-aaaaaaaa"
    newer = base / "20250102T000000-bbbbbbbb"
    older.mkdir()
    newer.mkdir()
    _write_complete_manifest(older)
    _write_complete_manifest(newer)
    (base / "CURRENT").write_text(older.name, encoding="utf-8")

    snap.recover_snapshots(base)
    assert (base / "CURRENT").read_text(encoding="utf-8").strip() == older.name


def test_recover_snapshots_removes_stale_artifacts_without_inference(
    tmp_path: Path,
) -> None:
    base = tmp_path / "storage"
    base.mkdir()
    (base / "_tmp-stale").mkdir()
    (base / ".lock.stale-1").write_text("x", encoding="utf-8")
    (base / ".lock.meta.json.stale-1").write_text("x", encoding="utf-8")

    v1 = base / "20250101T000000-aaaaaaaa"
    v2 = base / "20250102T000000-bbbbbbbb"
    v1.mkdir()
    v2.mkdir()
    _write_complete_manifest(v1)
    _write_complete_manifest(v2)

    snap.recover_snapshots(base)

    assert not (base / "_tmp-stale").exists()
    assert not (base / ".lock.stale-1").exists()
    assert not (base / ".lock.meta.json.stale-1").exists()
    assert not (base / "CURRENT").exists()
    assert v1.is_dir()
    assert v2.is_dir()


def test_recovery_removes_dangling_current_symlink(tmp_path: Path) -> None:
    """A dangling pointer is invalid state, not an absent CURRENT file."""
    base = tmp_path / "storage"
    base.mkdir()
    current = base / "CURRENT"
    current.symlink_to(base / "missing")

    snap.recover_snapshots(base)

    assert not current.exists()
    assert not current.is_symlink()


def test_corrupt_current_is_discarded_without_fallback(tmp_path: Path) -> None:
    """Readers and recovery never infer a replacement for corrupt CURRENT."""
    base = tmp_path / "storage"
    base.mkdir()
    older = base / "20250101T000000-aaaaaaaa"
    newer = base / "20250102T000000-bbbbbbbb"
    for directory in (older, newer):
        directory.mkdir()
        (directory / "payload.txt").write_text(directory.name, encoding="utf-8")
        _write_complete_manifest(directory)
    (newer / "payload.txt").write_text("tampered", encoding="utf-8")
    (base / "CURRENT").write_text(newer.name, encoding="utf-8")

    assert snap.load_manifest(newer) is None
    assert snap.latest_snapshot_dir(base) is None

    snap.recover_snapshots(base)
    assert not (base / "CURRENT").exists()
    assert older.is_dir()


def test_recover_snapshots_ignores_incomplete_final_directory(tmp_path: Path) -> None:
    """Recovery never repairs CURRENT to a final-named incomplete snapshot."""
    base = tmp_path / "storage"
    base.mkdir()
    complete = base / "20250101T000000-aaaaaaaa"
    incomplete = base / "20250102T000000-bbbbbbbb"
    complete.mkdir()
    incomplete.mkdir()
    _write_complete_manifest(complete)
    (incomplete / "manifest.meta.json").write_text(
        '{"complete":false}', encoding="utf-8"
    )
    (base / "CURRENT").write_text(incomplete.name, encoding="utf-8")

    snap.recover_snapshots(base)

    assert not (base / "CURRENT").exists()
    assert complete.is_dir()


def test_recovery_removes_unreferenced_invalid_final_named_crash_debris(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed post-rename final directory is removed without touching unknown dirs."""
    base = tmp_path / "storage"
    base.mkdir()
    invalid = base / "20250102T000000-deadbeef"
    invalid.mkdir()
    (invalid / "manifest.meta.json").write_text("{", encoding="utf-8")
    unknown = base / "operator-notes"
    unknown.mkdir()
    monkeypatch.setattr(snap.settings.snapshots, "gc_grace_seconds", 0)

    snap.recover_snapshots(base)

    assert not invalid.exists()
    assert unknown.is_dir()


def test_recovery_preserves_current_and_never_promotes_newer_snapshot(
    tmp_path: Path,
) -> None:
    """A complete unreferenced snapshot cannot supersede CURRENT."""
    base = tmp_path / "storage"
    base.mkdir()
    current = base / "20250101T000000-aaaaaaaa"
    unreferenced = base / "20250102T000000-bbbbbbbb"
    for directory in (current, unreferenced):
        directory.mkdir()
        _write_complete_manifest(directory)
    (base / "CURRENT").write_text(current.name, encoding="utf-8")

    snap.recover_snapshots(base)

    assert snap.latest_snapshot_dir(base) == current
    assert (base / "CURRENT").read_text(encoding="utf-8") == current.name
    assert unreferenced.is_dir()


def test_recovery_discards_journaled_uncommitted_destination(tmp_path: Path) -> None:
    """A hard crash before CURRENT cannot pin a failed complete generation."""
    base = tmp_path / "storage"
    base.mkdir()
    failed = base / "20250102T000000-bbbbbbbb"
    failed.mkdir()
    _write_complete_manifest(failed)
    (base / ".activation-transaction.json").write_text(
        '{"destination":"20250102T000000-bbbbbbbb","schema_version":1}',
        encoding="utf-8",
    )

    snap.recover_snapshots(base)

    assert not failed.exists()
    assert not (base / ".activation-transaction.json").exists()


def test_recovery_retires_journal_after_committed_current(tmp_path: Path) -> None:
    """A crash after CURRENT only leaves harmless journal retirement work."""
    base = tmp_path / "storage"
    base.mkdir()
    current = base / "20250102T000000-bbbbbbbb"
    current.mkdir()
    _write_complete_manifest(current)
    (base / "CURRENT").write_text(current.name, encoding="utf-8")
    (base / ".activation-transaction.json").write_text(
        '{"destination":"20250102T000000-bbbbbbbb","schema_version":1}',
        encoding="utf-8",
    )

    snap.recover_snapshots(base)

    assert snap.latest_snapshot_dir(base) == current
    assert not (base / ".activation-transaction.json").exists()


def test_persist_graph_storage_context_uses_native_file_contract(
    tmp_path: Path,
) -> None:
    from llama_index.core import StorageContext
    from llama_index.core.graph_stores import SimplePropertyGraphStore
    from llama_index.core.graph_stores.types import DEFAULT_PG_PERSIST_FNAME

    workspace = tmp_path / "_tmp-graph"
    out_dir = workspace / "graph"
    workspace.mkdir()

    context = StorageContext.from_defaults(
        property_graph_store=SimplePropertyGraphStore()
    )
    snap.persist_graph_storage_context(context, out_dir)

    assert {path.name for path in out_dir.iterdir()} == {
        "default__vector_store.json",
        "docstore.json",
        "graph_store.json",
        "image__vector_store.json",
        "index_store.json",
        DEFAULT_PG_PERSIST_FNAME,
    }
    loaded = StorageContext.from_defaults(persist_dir=str(out_dir))
    assert isinstance(loaded.property_graph_store, SimplePropertyGraphStore)
    assert loaded.vector_store is not None


def test_loaders_return_none_when_snapshot_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        snap, "latest_snapshot_dir", lambda *_a, **_k: None, raising=True
    )
    assert snap.load_vector_index(None) is None

    # Ensure PropertyGraphIndex imports succeed even when graphs are unused.
    core_mod = ModuleType("llama_index.core")
    graph_mod = ModuleType("llama_index.core.graph_stores")
    core_mod.PropertyGraphIndex = object()  # type: ignore[attr-defined]
    graph_mod.SimplePropertyGraphStore = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index.core", core_mod)
    monkeypatch.setitem(sys.modules, "llama_index.core.graph_stores", graph_mod)
    assert snap.load_property_graph_index(None) is None


def test_loaders_return_none_when_payload_dirs_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    _write_complete_manifest(snapshot_dir)
    monkeypatch.setattr(
        snap, "latest_snapshot_dir", lambda *_a, **_k: snapshot_dir, raising=True
    )

    assert snap.load_vector_index(None) is None

    core_mod = ModuleType("llama_index.core")
    graph_mod = ModuleType("llama_index.core.graph_stores")
    core_mod.PropertyGraphIndex = object()  # type: ignore[attr-defined]
    graph_mod.SimplePropertyGraphStore = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index.core", core_mod)
    monkeypatch.setitem(sys.modules, "llama_index.core.graph_stores", graph_mod)
    assert snap.load_property_graph_index(None) is None


def test_index_loaders_ignore_explicit_incomplete_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit index loads cannot bypass the complete-manifest gate."""
    snapshot_dir = tmp_path / "snap"
    (snapshot_dir / "vector").mkdir(parents=True)
    (snapshot_dir / "graph").mkdir()
    (snapshot_dir / "manifest.meta.json").write_text(
        '{"complete":false}', encoding="utf-8"
    )

    core_mod = ModuleType("llama_index.core")
    graph_mod = ModuleType("llama_index.core.graph_stores")

    class _UnexpectedStorageContext:
        @classmethod
        def from_defaults(cls, *, persist_dir: str) -> None:
            raise AssertionError(f"unexpected vector load from {persist_dir}")

    class _UnexpectedGraphStore:
        @classmethod
        def from_persist_dir(cls, persist_dir: str) -> None:
            raise AssertionError(f"unexpected graph load from {persist_dir}")

    core_mod.StorageContext = _UnexpectedStorageContext  # type: ignore[attr-defined]
    core_mod.load_index_from_storage = object()  # type: ignore[attr-defined]
    core_mod.PropertyGraphIndex = object()  # type: ignore[attr-defined]
    graph_mod.SimplePropertyGraphStore = _UnexpectedGraphStore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index.core", core_mod)
    monkeypatch.setitem(sys.modules, "llama_index.core.graph_stores", graph_mod)

    assert snap.load_vector_index(snapshot_dir) is None
    assert snap.load_property_graph_index(snapshot_dir) is None
