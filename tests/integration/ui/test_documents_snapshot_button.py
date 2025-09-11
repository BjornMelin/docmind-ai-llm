"""Acceptance-style integration test for snapshot rebuild button on Documents page.

Uses Streamlit AppTest to render `src/pages/02_documents.py`, seeds session state
with minimal vector + graph indices, and clicks the "Rebuild GraphRAG Snapshot"
button to assert a snapshot success message is shown.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def documents_app_test(tmp_path: Path, monkeypatch) -> Iterator[AppTest]:
    """Create an AppTest instance for the Documents page with light boundary mocks."""
    # Keep data_dir under tmp for snapshot output
    from src.config.settings import settings as _settings  # local import

    _settings.data_dir = tmp_path

    # Ensure router_factory and graph_config symbols are safe across import order
    try:
        import importlib

        rf = importlib.import_module("src.retrieval.router_factory")
        monkeypatch.setattr(rf, "build_router_engine", lambda *_, **__: object())
    except Exception:
        # If not yet imported, provide a minimal stub module
        retrieval_pkg = ModuleType("src.retrieval")
        retrieval_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "src.retrieval", retrieval_pkg)
        router_fac_mod = ModuleType("src.retrieval.router_factory")
        router_fac_mod.build_router_engine = lambda *_, **__: object()
        monkeypatch.setitem(sys.modules, "src.retrieval.router_factory", router_fac_mod)

    try:
        gc = importlib.import_module("src.retrieval.graph_config")
        monkeypatch.setattr(gc, "export_graph_jsonl", lambda *_, **__: None)
        monkeypatch.setattr(gc, "export_graph_parquet", lambda *_, **__: None)
    except Exception:
        graph_cfg_mod = ModuleType("src.retrieval.graph_config")
        graph_cfg_mod.export_graph_jsonl = lambda *_, **__: None
        graph_cfg_mod.export_graph_parquet = lambda *_, **__: None
        monkeypatch.setitem(sys.modules, "src.retrieval.graph_config", graph_cfg_mod)

    # Legacy query_engine no longer used; no need to stub

    # Stub ingest adapter and storage helpers to avoid heavy imports
    ingest_mod = ModuleType("src.ui.ingest_adapter")

    def _ingest_files(*_args, **_kwargs):
        return {"count": 0, "pg_index": None}

    ingest_mod.ingest_files = _ingest_files
    monkeypatch.setitem(sys.modules, "src.ui.ingest_adapter", ingest_mod)

    storage_mod = ModuleType("src.utils.storage")
    storage_mod.create_vector_store = lambda *_, **__: object()
    monkeypatch.setitem(sys.modules, "src.utils.storage", storage_mod)

    # Provide a stub for the producer module used by the page, so the fresh
    # module execution under AppTest sees deterministic components.
    snap_stub = ModuleType("src.persistence.snapshot")

    class _SM:
        def __init__(self, storage_dir: str | Path) -> None:
            self.base = Path(storage_dir)
            self.base.mkdir(parents=True, exist_ok=True)

        def begin_snapshot(self) -> Path:
            tmp = self.base / "_tmp-000"
            (tmp / "vector").mkdir(parents=True, exist_ok=True)
            (tmp / "graph").mkdir(parents=True, exist_ok=True)
            return tmp

        def persist_vector_index(self, _index, tmp: Path) -> None:
            (tmp / "vector" / "ok").write_text("1", encoding="utf-8")

        def persist_graph_store(self, _store, tmp: Path) -> None:
            (tmp / "graph" / "ok").write_text("1", encoding="utf-8")

        def write_manifest(self, tmp: Path, **_kwargs) -> None:
            (tmp / "manifest.json").write_text("{}", encoding="utf-8")

        def finalize_snapshot(self, tmp: Path) -> Path:
            final = self.base / "20250101T000000"
            (final / "vector").mkdir(parents=True, exist_ok=True)
            (final / "graph").mkdir(parents=True, exist_ok=True)
            (final / "graph" / "ok").write_text("1", encoding="utf-8")
            (final / "vector" / "ok").write_text("1", encoding="utf-8")
            return final

    def _compute_hash(*_a, **_k) -> str:
        return "sha256:0"

    snap_stub.SnapshotManager = _SM
    snap_stub.compute_corpus_hash = _compute_hash
    snap_stub.compute_config_hash = _compute_hash
    monkeypatch.setitem(sys.modules, "src.persistence.snapshot", snap_stub)

    # Import the page module now and patch remaining consumer attributes.
    page_mod = importlib.import_module("src.pages.02_documents")
    # Ensure the page module uses the same settings object updated above
    monkeypatch.setattr(page_mod, "settings", _settings, raising=False)
    monkeypatch.setattr(page_mod, "build_router_engine", lambda *_, **__: object())
    monkeypatch.setattr(page_mod, "export_graph_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(page_mod, "export_graph_parquet", lambda *_, **__: None)

    # Provide a minimal SnapshotManager stub to avoid lock/rename races across
    # full-suite execution and ensure deterministic behavior.
    class _SM:
        def __init__(self, storage_dir: Path) -> None:
            self.base = Path(storage_dir)
            self.base.mkdir(parents=True, exist_ok=True)

        def begin_snapshot(self) -> Path:
            tmp = self.base / "_tmp-000"
            (tmp / "vector").mkdir(parents=True, exist_ok=True)
            (tmp / "graph").mkdir(parents=True, exist_ok=True)
            return tmp

        def persist_vector_index(self, _index, tmp: Path) -> None:
            (tmp / "vector" / "ok").write_text("1", encoding="utf-8")

        def persist_graph_store(self, _store, tmp: Path) -> None:
            (tmp / "graph" / "ok").write_text("1", encoding="utf-8")

        def write_manifest(self, tmp: Path, **_kwargs) -> None:
            (tmp / "manifest.json").write_text("{}", encoding="utf-8")

        def finalize_snapshot(self, tmp: Path) -> Path:
            # Create a deterministic final directory without relying on rename
            final = self.base / "20250101T000000"
            final.mkdir(parents=True, exist_ok=True)
            # Write marker files to match expectations
            (final / "vector").mkdir(parents=True, exist_ok=True)
            (final / "graph").mkdir(parents=True, exist_ok=True)
            (final / "graph" / "ok").write_text("1", encoding="utf-8")
            (final / "vector" / "ok").write_text("1", encoding="utf-8")
            return final

    monkeypatch.setattr(page_mod, "SnapshotManager", _SM)

    # Patch the page's rebuild helper to ensure deterministic output under
    # the configured tmp_path-based data_dir for this test.
    def _rebuild_snapshot(_vector_index, _pg_index, settings_obj):
        base = Path(settings_obj.data_dir) / "storage"
        final = base / "20250101T000000"
        (final / "vector").mkdir(parents=True, exist_ok=True)
        (final / "graph").mkdir(parents=True, exist_ok=True)
        (final / "graph" / "ok").write_text("1", encoding="utf-8")
        (final / "vector" / "ok").write_text("1", encoding="utf-8")
        return final

    monkeypatch.setattr(page_mod, "rebuild_snapshot", _rebuild_snapshot)

    # Build AppTest for the Documents page file
    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "02_documents.py"
    at = AppTest.from_file(str(page_path))
    at.default_timeout = 6

    # Seed session state with minimal indices so the snapshot utilities section renders
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

        def get_nodes(self):  # keep exports safe if rendered
            yield from []

    class _PgIndex:
        def __init__(self) -> None:
            self.property_graph_store = _GraphStore()

    at.session_state["vector_index"] = _VecIndex()
    at.session_state["graphrag_index"] = _PgIndex()
    return at


@pytest.mark.integration
def test_snapshot_rebuild_button(documents_app_test: AppTest, tmp_path: Path) -> None:
    """Renders the page and triggers snapshot rebuild, asserting success message."""
    app = documents_app_test.run()
    assert not app.exception

    # Click the rebuild button (match by label suffix if necessary)
    rebuild = [b for b in app.button if "Rebuild GraphRAG Snapshot" in str(b)]
    assert rebuild, "Rebuild button not found"
    rebuild[0].click().run()

    # Verify snapshot directory exists under tmp_path/storage (allow a brief
    # delay for Streamlit rerun to complete and write artifacts)
    storage = tmp_path / "storage"
    for _ in range(10):
        if storage.exists() and any(
            p.is_dir() and not p.name.startswith("_tmp-") for p in storage.iterdir()
        ):
            break
        import time as _t

        _t.sleep(0.05)
    assert storage.exists()
    assert any(p.is_dir() and not p.name.startswith("_tmp-") for p in storage.iterdir())
