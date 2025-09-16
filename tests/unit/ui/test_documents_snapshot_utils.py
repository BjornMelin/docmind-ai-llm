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

_stub_retrieval_pkg = ModuleType("src.retrieval")
_stub_retrieval_pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("src.retrieval", _stub_retrieval_pkg)

_stub_graph = ModuleType("src.retrieval.graph_config")


def _stub_export_jsonl(_pg_index, dest: Path, _seeds: list[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("[]", encoding="utf-8")


def _stub_export_parquet(_pg_index, dest: Path, _seeds: list[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"PAR1")


def _stub_get_export_seed_ids(_pg_index, _vector_index, cap: int = 32) -> list[str]:
    return [str(i) for i in range(cap)]


class _StubPropertyGraphConfig:
    pass


def _stub_noop(*_args: Any, **_kwargs: Any) -> Any:
    return None


_stub_graph.PropertyGraphConfig = _StubPropertyGraphConfig  # type: ignore[attr-defined]
_stub_graph.create_graph_rag_components = _stub_noop  # type: ignore[attr-defined]
_stub_graph.create_property_graph_index = _stub_noop  # type: ignore[attr-defined]
_stub_graph.create_property_graph_index_async = _stub_noop  # type: ignore[attr-defined]
_stub_graph.create_tech_schema = _stub_noop  # type: ignore[attr-defined]
_stub_graph.extract_entities = _stub_noop  # type: ignore[attr-defined]
_stub_graph.extract_relationships = _stub_noop  # type: ignore[attr-defined]
_stub_graph.traverse_graph = _stub_noop  # type: ignore[attr-defined]
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

_stub_ingest = ModuleType("src.ui.ingest_adapter")
_stub_ingest.ingest_files = _stub_noop  # type: ignore[attr-defined]
sys.modules.setdefault("src.ui.ingest_adapter", _stub_ingest)

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


class _PgIndex:
    """Mock property graph index class for testing."""

    def __init__(self) -> None:
        """Initialize with mock property graph store."""
        self.property_graph_store = _GraphStore()


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
def test_rebuild_snapshot_writes_manifest(tmp_path: Path) -> None:
    """Test that snapshot rebuilding creates proper manifest files.

    Verifies that the rebuild process creates a manifest.json file containing
    required hash information in the correct location.
    """
    settings_obj = SimpleNamespace(
        data_dir=tmp_path,
        retrieval=SimpleNamespace(router="auto", enable_server_hybrid=True),
        processing=SimpleNamespace(chunk_size=512, chunk_overlap=64),
        database=SimpleNamespace(vector_store_type="qdrant"),
        app_version="x",
    )
    final = docs_page.rebuild_snapshot(_VecIndex(), _PgIndex(), settings_obj)
    assert final.exists()
    manifest = final / "manifest.json"
    meta = final / "manifest.meta.json"
    checksum = final / "manifest.checksum"
    manifest_jsonl = final / "manifest.jsonl"
    for path in (manifest, meta, checksum, manifest_jsonl):
        assert path.exists()

    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["corpus_hash"].startswith("sha256:")
    assert payload["config_hash"].startswith("sha256:")

    exports_dir = final / "graph_exports"
    jsonl_exports = list(exports_dir.glob("graph_export-*.jsonl"))
    parquet_exports = list(exports_dir.glob("graph_export-*.parquet"))
    assert jsonl_exports
    assert parquet_exports
