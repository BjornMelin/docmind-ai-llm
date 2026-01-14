"""Targeted unit tests for snapshot verification and recovery helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

import src.persistence.snapshot as snap
from src.persistence import snapshot_writer as writer

pytestmark = pytest.mark.unit


def test_verify_snapshot_roundtrip_and_mismatch(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    payload = snapshot_dir / "payload.txt"
    payload.write_text("hello", encoding="utf-8")

    writer.write_manifest(snapshot_dir, manifest_meta={"index_id": "x"})
    assert snap.verify_snapshot(snapshot_dir) is True

    payload.write_text("tampered", encoding="utf-8")
    assert snap.verify_snapshot(snapshot_dir) is False


def test_recover_snapshots_keeps_current_when_valid(tmp_path: Path) -> None:
    base = tmp_path / "storage"
    base.mkdir()
    older = base / "20250101T000000-aaaa"
    newer = base / "20250102T000000-bbbb"
    older.mkdir()
    newer.mkdir()
    (base / "CURRENT").write_text(older.name, encoding="utf-8")

    snap.recover_snapshots(base)
    assert (base / "CURRENT").read_text(encoding="utf-8").strip() == older.name


def test_recover_snapshots_repairs_current_and_removes_stale_artifacts(
    tmp_path: Path,
) -> None:
    base = tmp_path / "storage"
    base.mkdir()
    (base / "_tmp-stale").mkdir()
    (base / ".lock.stale-1").write_text("x", encoding="utf-8")
    (base / ".lock.meta.json.stale-1").write_text("x", encoding="utf-8")

    v1 = base / "20250101T000000-aaaa"
    v2 = base / "20250102T000000-bbbb"
    v1.mkdir()
    v2.mkdir()

    snap.recover_snapshots(base)

    assert not (base / "_tmp-stale").exists()
    assert not (base / ".lock.stale-1").exists()
    assert not (base / ".lock.meta.json.stale-1").exists()
    assert (base / "CURRENT").read_text(encoding="utf-8").strip() == v2.name


def test_persist_graph_store_falls_back_to_positional_persist(tmp_path: Path) -> None:
    workspace = tmp_path / "_tmp-graph"
    out_dir = workspace / "graph"
    workspace.mkdir()

    called: list[str] = []

    # Force TypeError on keyword invocation
    class _PositionalNoKw:
        def persist(self, path: str) -> None:  # type: ignore[no-untyped-def]
            called.append(path)

    snap.persist_graph_store(_PositionalNoKw(), out_dir)
    assert any(Path(p).name == "graph" for p in called)


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
