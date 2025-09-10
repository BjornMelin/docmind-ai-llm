"""Unit tests for SnapshotManager (SPEC-014)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from src.persistence.snapshot import (
    SnapshotManager,
    compute_config_hash,
    compute_corpus_hash,
)


class _DummyStorage:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def persist(self, persist_dir: str) -> None:
        # Write a marker file to signal persist occurred
        out = Path(persist_dir) / "storage.ok"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.marker, encoding="utf-8")


class _DummyIndex:
    def __init__(self) -> None:
        self.storage_context = _DummyStorage("ok")


class _DummyGraphStore:
    def persist(self, path: str) -> None:
        Path(path).write_text("{}", encoding="utf-8")


@pytest.mark.unit
def test_snapshot_manager_persists_and_finalizes(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    mgr = SnapshotManager(storage)
    paths = mgr.begin_snapshot()
    # Persist vector and graph
    mgr.persist_vector_index(_DummyIndex(), paths)
    mgr.persist_graph_store(_DummyGraphStore(), paths)
    # Write manifest
    mgr.write_manifest(
        paths,
        index_id="idx1",
        graph_store_type="simple",
        vector_store_type="qdrant",
        corpus_hash="abc",
        config_hash="def",
        versions={"llama_index": "0.13.4"},
    )
    final = mgr.finalize_snapshot(paths)

    assert final.exists()
    assert (final / "vector" / "storage.ok").exists()
    assert (final / "graph" / "graph_store.json").exists()

    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["index_id"] == "idx1"
    assert manifest["graph_store_type"] == "simple"
    assert manifest["vector_store_type"] == "qdrant"
    assert manifest["corpus_hash"] == "abc"
    assert manifest["config_hash"] == "def"
    assert "created_at" in manifest
    assert "versions" in manifest


@pytest.mark.unit
def test_snapshot_manager_lock_contention(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    mgr1 = SnapshotManager(storage)
    paths1 = mgr1.begin_snapshot()

    mgr2 = SnapshotManager(storage)
    # Tighten lock timeout for the second manager to force a quick failure
    mgr2._lock.timeout_s = 0.2  # type: ignore[attr-defined]
    mgr2._lock.poll_interval_s = 0.01  # type: ignore[attr-defined]
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        mgr2.begin_snapshot()
    assert time.monotonic() - start < 1.5

    # cleanup lock by finalizing the first snapshot
    mgr1.write_manifest(paths1, index_id="a")
    mgr1.finalize_snapshot(paths1)


@pytest.mark.unit
def test_hashing_stability(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello", encoding="utf-8")
    b.write_text("world", encoding="utf-8")
    # Set stable mtimes with ns
    os.utime(a, ns=(int(time.time_ns()), int(time.time_ns())))
    os.utime(b, ns=(int(time.time_ns()), int(time.time_ns())))

    h1 = compute_corpus_hash([a, b])
    h2 = compute_corpus_hash([b, a])
    assert h1 == h2

    # Change content -> size changes -> hash changes
    b.write_text("world+", encoding="utf-8")
    h3 = compute_corpus_hash([a, b])
    assert h3 != h1

    cfg1 = {"a": 1, "b": [2, 3]}
    cfg2 = {"b": [2, 3], "a": 1}  # different order
    assert compute_config_hash(cfg1) == compute_config_hash(cfg2)
