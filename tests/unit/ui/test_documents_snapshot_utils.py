"""Unit tests for Documents page helpers: seed cap and rebuild snapshot.

This module contains unit tests for utility functions related to document snapshots,
including seed ID collection with capacity limits and snapshot rebuilding functionality.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.retrieval.graph_config import get_export_seed_ids

docs_page = importlib.import_module("src.pages.02_documents")


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
        retrieval=SimpleNamespace(router="auto", hybrid_enabled=True),
        processing=SimpleNamespace(chunk_size=512, chunk_overlap=64),
        database=SimpleNamespace(vector_store_type="qdrant"),
        app_version="x",
    )
    final = docs_page.rebuild_snapshot(_VecIndex(), _PgIndex(), settings_obj)
    assert final.exists()
    manifest = final / "manifest.json"
    assert manifest.exists()
    txt = manifest.read_text(encoding="utf-8")
    assert "corpus_hash" in txt
    assert "config_hash" in txt
