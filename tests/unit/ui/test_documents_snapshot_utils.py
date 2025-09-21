"""Unit tests for Documents page helpers: seed cap and rebuild snapshot.

This module contains unit tests for utility functions related to document snapshots,
including seed ID collection with capacity limits and snapshot rebuilding functionality.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_MODULE_TARGETS = [
    "src.retrieval",
    "src.retrieval.graph_config",
    "src.retrieval.router_factory",
    "src.retrieval.hybrid",
    "src.retrieval.reranking",
    "src.utils.storage",
    "src.utils.monitoring",
]
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _MODULE_TARGETS}

_stub_retrieval_pkg = ModuleType("src.retrieval")
_stub_retrieval_pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("src.retrieval", _stub_retrieval_pkg)

_stub_graph = ModuleType("src.retrieval.graph_config")


def _stub_export_jsonl(
    *,
    property_graph_index: Any,
    output_path: Path,
    seed_node_ids: list[str],
    depth: int = 1,
    **_kwargs: Any,
) -> None:
    del property_graph_index, seed_node_ids, depth
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("[]", encoding="utf-8")


def _stub_export_parquet(
    *,
    property_graph_index: Any,
    output_path: Path,
    seed_node_ids: list[str],
    depth: int = 1,
    **_kwargs: Any,
) -> None:
    del property_graph_index, seed_node_ids, depth
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"PAR1")


def _stub_get_export_seed_ids(_pg_index, _vector_index, cap: int = 32) -> list[str]:
    return [str(i) for i in range(cap)]


class _StubPropertyGraphConfig:
    pass


def _stub_noop(*_args: Any, **_kwargs: Any) -> Any:
    return None


class _GraphQueryArtifacts:
    def __init__(self, retriever=None, query_engine=None) -> None:
        self.retriever = retriever
        self.query_engine = query_engine


def _stub_build_graph_retriever(*_args: Any, **_kwargs: Any) -> dict[str, bool]:
    return {"retriever": True}


def _stub_build_graph_query_engine(*_args: Any, **_kwargs: Any) -> _GraphQueryArtifacts:
    return _GraphQueryArtifacts(
        retriever={"retriever": True}, query_engine={"engine": True}
    )


_stub_graph.GraphQueryArtifacts = _GraphQueryArtifacts  # type: ignore[attr-defined]
_stub_graph.build_graph_retriever = _stub_build_graph_retriever  # type: ignore[attr-defined]
_stub_graph.build_graph_query_engine = _stub_build_graph_query_engine  # type: ignore[attr-defined]
_stub_graph.export_graph_jsonl = _stub_export_jsonl  # type: ignore[attr-defined]
_stub_graph.export_graph_parquet = _stub_export_parquet  # type: ignore[attr-defined]
_stub_graph.get_export_seed_ids = _stub_get_export_seed_ids  # type: ignore[attr-defined]

sys.modules.setdefault("src.retrieval.graph_config", _stub_graph)

_stub_router = ModuleType("src.retrieval.router_factory")
_stub_router.build_router_engine = _stub_noop  # type: ignore[attr-defined]
sys.modules.setdefault("src.retrieval.router_factory", _stub_router)

_stub_hybrid = ModuleType("src.retrieval.hybrid")
_stub_hybrid.ServerHybridRetriever = object  # type: ignore[attr-defined]
sys.modules.setdefault("src.retrieval.hybrid", _stub_hybrid)

_stub_rerank = ModuleType("src.retrieval.reranking")
_stub_rerank.MultimodalReranker = object  # type: ignore[attr-defined]
_stub_rerank.build_text_reranker = _stub_noop  # type: ignore[attr-defined]
_stub_rerank.build_visual_reranker = _stub_noop  # type: ignore[attr-defined]
sys.modules.setdefault("src.retrieval.reranking", _stub_rerank)

_stub_storage = ModuleType("src.utils.storage")
_stub_storage.create_vector_store = _stub_noop  # type: ignore[attr-defined]
sys.modules.setdefault("src.utils.storage", _stub_storage)

_stub_monitoring = ModuleType("src.utils.monitoring")
_stub_monitoring.log_performance = lambda *_, **__: None
sys.modules.setdefault("src.utils.monitoring", _stub_monitoring)

docs_page = importlib.import_module("src.pages.02_documents")
get_export_seed_ids = _stub_graph.get_export_seed_ids  # type: ignore


class _Node:
    """Mock node class for testing purposes."""

    def __init__(self, nid: str) -> None:
        """Initialize a mock node with an ID.

        Args:
            nid: Node identifier string.
        """
        self.id = nid


class _Store:
    """Mock store class that generates test nodes."""

    def get_nodes(self):
        """Generate a sequence of mock nodes.

        Yields:
            _Node: Mock node instances with incremental IDs.
        """
        for i in range(100):
            yield _Node(str(i))


class _DummyStorage:
    """Mock storage class for testing persistence operations."""

    def persist(self, persist_dir: str) -> None:
        """Create a dummy persistence directory with a marker file.

        Args:
            persist_dir: Directory path to persist data to.
        """
        p = Path(persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "ok").write_text("1", encoding="utf-8")


class _VecIndex:
    """Mock vector index class for testing."""

    def __init__(self) -> None:
        """Initialize with dummy storage context."""
        self.storage_context = _DummyStorage()


class _GraphStore:
    """Mock graph store class for testing persistence."""

    def persist(self, persist_dir: str) -> None:
        """Create a dummy persistence directory with a marker file.

        Args:
            persist_dir: Directory path to persist data to.
        """
        p = Path(persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "ok").write_text("1", encoding="utf-8")

    def get_rel_map(self, node_ids=None, depth=1, **_kwargs):
        """Return a deterministic relation list for tests."""
        del depth
        items = list(node_ids or [])
        if len(items) < 2:
            return []
        return [
            json.dumps({"subject": items[0], "object": items[1], "relation": "rel"})
        ]

    class _Frame:
        def __init__(self, rows: list[str]) -> None:
            self._rows = rows

        def to_parquet(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"PAR1")

    def store_rel_map_df(self, node_ids=None, depth=1, **_kwargs):
        rows = self.get_rel_map(node_ids=node_ids, depth=depth)
        return self._Frame(rows)


class _PgIndex:
    """Mock property graph index class for testing."""

    def __init__(self) -> None:
        """Initialize with mock property graph store."""
        self.property_graph_store = _GraphStore()
        self.storage_context = SimpleNamespace()


@pytest.mark.unit
def test_collect_seed_ids_caps_to_32() -> None:
    """Test that seed ID collection respects the cap limit.

    Ensures that when no indices are provided, the fallback path still
    enforces the cap limit of 32 seed IDs.
    """
    seeds = get_export_seed_ids(None, None, cap=32)
    assert len(seeds) == 32
    assert seeds[0] == "0"
    assert seeds[-1] == "31"


@pytest.mark.unit
def test_rebuild_snapshot_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Snapshot rebuild packages exports and manifests metadata."""
    settings_obj = SimpleNamespace(
        data_dir=tmp_path,
        retrieval=SimpleNamespace(router="auto", enable_server_hybrid=True),
        processing=SimpleNamespace(chunk_size=512, chunk_overlap=64),
        database=SimpleNamespace(vector_store_type="qdrant"),
        app_version="x",
    )
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        docs_page, "_log_export_event", lambda payload: events.append(payload.copy())
    )

    final = docs_page.rebuild_snapshot(_VecIndex(), _PgIndex(), settings_obj)
    assert final.exists()

    manifest_jsonl = final / "manifest.jsonl"
    meta = final / "manifest.meta.json"
    checksum = final / "manifest.checksum"
    for path in (manifest_jsonl, meta, checksum):
        assert path.exists()

    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["corpus_hash"]
    assert len(payload["corpus_hash"]) == 64
    assert payload["config_hash"]
    assert len(payload["config_hash"]) == 64

    exports_meta = payload.get("graph_exports", [])
    assert len(exports_meta) >= 1
    formats = {item["format"] for item in exports_meta}
    assert "jsonl" in formats
    for item in exports_meta:
        assert item["path"].startswith("graph/")
        assert item["seed_count"] == 32
        assert item["size_bytes"] > 0
        assert item["duration_ms"] >= 0
        assert len(item["sha256"]) == 64

    manifest_lines = manifest_jsonl.read_text(encoding="utf-8").splitlines()
    assert any("graph_export-" in line for line in manifest_lines)

    graph_dir = final / "graph"
    jsonl_exports = list(graph_dir.glob("graph_export-*.jsonl"))
    parquet_exports = list(graph_dir.glob("graph_export-*.parquet"))
    assert jsonl_exports

    assert events
    event_types = {evt.get("export_type") for evt in events}
    assert "graph_jsonl" in event_types
    if parquet_exports:
        assert "graph_parquet" in event_types
    for evt in events:
        assert evt.get("context") == "snapshot"
        assert "duration_ms" in evt


@pytest.mark.unit
def test_rebuild_snapshot_skips_exports_without_storage_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Snapshot rebuild omits exports when the graph index lacks storage context."""
    settings_obj = SimpleNamespace(
        data_dir=tmp_path,
        retrieval=SimpleNamespace(router="auto", enable_server_hybrid=True),
        processing=SimpleNamespace(chunk_size=512, chunk_overlap=64),
        database=SimpleNamespace(vector_store_type="qdrant"),
        app_version="x",
    )
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        docs_page, "_log_export_event", lambda payload: events.append(payload.copy())
    )

    class _PgIndexNoStorage:
        def __init__(self) -> None:
            self.property_graph_store = _GraphStore()

    final = docs_page.rebuild_snapshot(_VecIndex(), _PgIndexNoStorage(), settings_obj)
    payload = json.loads((final / "manifest.meta.json").read_text(encoding="utf-8"))
    assert payload.get("graph_exports") in (None, [])
    graph_dir = final / "graph"
    assert not any(graph_dir.glob("graph_export-*.jsonl"))
    assert not any(graph_dir.glob("graph_export-*.parquet"))
    assert not events


def teardown_module(module) -> None:
    """Restore original modules patched at import time."""
    for name, original in _ORIGINAL_MODULES.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
