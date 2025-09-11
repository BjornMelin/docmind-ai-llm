"""Unit tests for Documents page helpers: seed cap and rebuild snapshot."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

docs_page = importlib.import_module("src.pages.02_documents")
from src.retrieval.graph_config import get_export_seed_ids


class _Node:
    def __init__(self, nid: str) -> None:
        self.id = nid


class _Store:
    def get_nodes(self):
        for i in range(100):
            yield _Node(str(i))


class _DummyStorage:
    def persist(self, persist_dir: str) -> None:
        p = Path(persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "ok").write_text("1", encoding="utf-8")


class _VecIndex:
    def __init__(self) -> None:
        self.storage_context = _DummyStorage()


class _GraphStore:
    def persist(self, persist_dir: str) -> None:
        p = Path(persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "ok").write_text("1", encoding="utf-8")


class _PgIndex:
    def __init__(self) -> None:
        self.property_graph_store = _GraphStore()


@pytest.mark.unit
def test_collect_seed_ids_caps_to_32() -> None:
    # Using fallback path (no indices), cap enforcement still holds
    seeds = get_export_seed_ids(None, None, cap=32)
    assert len(seeds) == 32
    assert seeds[0] == "0"
    assert seeds[-1] == "31"


@pytest.mark.unit
def test_rebuild_snapshot_writes_manifest(tmp_path: Path) -> None:
    # Minimal settings-like structure
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
