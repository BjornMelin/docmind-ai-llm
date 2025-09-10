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

    # Provide a lightweight 'src.retrieval' package with minimal submodules
    retrieval_pkg = ModuleType("src.retrieval")
    retrieval_pkg.__path__ = []  # mark as package
    monkeypatch.setitem(sys.modules, "src.retrieval", retrieval_pkg)

    # Stub graph_config submodule with no-op exports
    graph_cfg_mod = ModuleType("src.retrieval.graph_config")
    graph_cfg_mod.export_graph_jsonl = lambda *_, **__: None
    graph_cfg_mod.export_graph_parquet = lambda *_, **__: None
    monkeypatch.setitem(sys.modules, "src.retrieval.graph_config", graph_cfg_mod)

    # Stub router_factory submodule
    router_fac_mod = ModuleType("src.retrieval.router_factory")
    router_fac_mod.build_router_engine = lambda *_, **__: object()
    monkeypatch.setitem(sys.modules, "src.retrieval.router_factory", router_fac_mod)

    # Stub query_engine submodule to avoid heavy imports
    qe_mod = ModuleType("src.retrieval.query_engine")

    class _DummyRetriever:
        def __init__(self, *_args, **_kwargs):
            pass

    class _HybridParams:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    def _create_engine(*_args, **_kwargs):
        return object()

    qe_mod.ServerHybridRetriever = _DummyRetriever
    qe_mod._HybridParams = _HybridParams  # type: ignore[attr-defined]
    qe_mod.create_adaptive_router_engine = _create_engine
    monkeypatch.setitem(sys.modules, "src.retrieval.query_engine", qe_mod)

    # Stub ingest adapter and storage helpers to avoid heavy imports
    ingest_mod = ModuleType("src.ui.ingest_adapter")

    def _ingest_files(*_args, **_kwargs):
        return {"count": 0, "pg_index": None}

    ingest_mod.ingest_files = _ingest_files
    monkeypatch.setitem(sys.modules, "src.ui.ingest_adapter", ingest_mod)

    storage_mod = ModuleType("src.utils.storage")
    storage_mod.create_vector_store = lambda *_, **__: object()
    monkeypatch.setitem(sys.modules, "src.utils.storage", storage_mod)

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
        def persist(self, path: str) -> None:
            Path(path).write_text("{}", encoding="utf-8")

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

    # Verify snapshot directory exists under tmp_path/storage
    storage = tmp_path / "storage"
    assert storage.exists()
    # Find at least one timestamped directory
    assert any(p.is_dir() and not p.name.startswith("_tmp-") for p in storage.iterdir())
