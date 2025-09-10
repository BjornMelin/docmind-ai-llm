"""Unit tests for snapshot persist_dir symmetry and round-trip loaders.

Verifies that graph store is persisted to a directory and that loader
functions use from_persist_dir/from_defaults appropriately (via stubs).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.persistence.snapshot import (
    SnapshotManager,
    load_property_graph_index,
    load_vector_index,
)


class _DummyVecStorage:
    def __init__(self, out: Path) -> None:
        self.out = out

    def persist(self, persist_dir: str) -> None:
        p = Path(persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "ok").write_text("1", encoding="utf-8")


class _VecIndex:
    def __init__(self) -> None:
        self.storage_context = _DummyVecStorage(Path("."))


class _GraphStore:
    def persist(self, persist_dir: str) -> None:
        p = Path(persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "ok").write_text("1", encoding="utf-8")


class _PgIndex:
    def __init__(self) -> None:
        self.property_graph_store = _GraphStore()


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point data_dir to tmp so SnapshotManager writes under it
    from src.config.settings import settings as _settings

    _settings.data_dir = tmp_path


def test_snapshot_roundtrip_with_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Persist a snapshot with our stub indices
    mgr = SnapshotManager(Path.cwd() / "unused")
    tmp = mgr.begin_snapshot()
    try:
        mgr.persist_vector_index(_VecIndex(), tmp)
        mgr.persist_graph_store(_PgIndex().property_graph_store, tmp)
        # Minimal manifest
        mgr.write_manifest(
            tmp,
            index_id="x",
            graph_store_type="property_graph",
            vector_store_type="qdrant",
            corpus_hash="sha256:0",
            config_hash="sha256:0",
            versions={"app": "test"},
        )
        final = mgr.finalize_snapshot(tmp)
    except Exception:
        mgr.cleanup_tmp(tmp)
        raise

    # Create stub modules for llama_index loaders
    core_mod = ModuleType("llama_index.core")
    graph_mod = ModuleType("llama_index.core.graph_stores")

    class _StorageContext:
        def __init__(self, persist_dir: str) -> None:
            self.persist_dir = persist_dir

        @classmethod
        def from_defaults(cls, persist_dir: str):  # type: ignore[override]
            return cls(persist_dir)

    def _load_index_from_storage(storage: _StorageContext):  # type: ignore[override]
        # Validate that the vector dir exists and contains the sentinel
        p = Path(storage.persist_dir) / "ok"
        assert p.exists()
        return SimpleNamespace(storage_dir=storage.persist_dir)

    class _PropertyGraphIndex:
        @staticmethod
        def from_existing(property_graph_store):  # type: ignore[override]
            return SimpleNamespace(property_graph_store=property_graph_store)

    class _SimplePropertyGraphStore:
        @staticmethod
        def from_persist_dir(persist_dir: str):  # type: ignore[override]
            # Validate sentinel exists
            p = Path(persist_dir) / "ok"
            assert p.exists()
            return SimpleNamespace(persist_dir=persist_dir)

    core_mod.StorageContext = _StorageContext
    core_mod.load_index_from_storage = _load_index_from_storage
    core_mod.PropertyGraphIndex = _PropertyGraphIndex
    graph_mod.SimplePropertyGraphStore = _SimplePropertyGraphStore

    monkeypatch.setitem(sys.modules, "llama_index.core", core_mod)
    monkeypatch.setitem(sys.modules, "llama_index.core.graph_stores", graph_mod)

    # Load via helpers and assert non-null
    vec = load_vector_index(final)
    pg = load_property_graph_index(final)
    assert vec is not None
    assert pg is not None
    # Spot-check attributes from stubs
    assert Path(vec.storage_dir).name == "vector"
    assert Path(pg.property_graph_store.persist_dir).name == "graph"
