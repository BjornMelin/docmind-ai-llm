"""Integration test validating the Documents page snapshot rebuild flow."""

from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from streamlit.testing.v1 import AppTest


def _install_stub(module_name: str, **attrs) -> None:
    """Register a lightweight stub module with the supplied attributes."""
    stub = ModuleType(module_name)
    for key, value in attrs.items():
        setattr(stub, key, value)
    sys.modules[module_name] = stub


@pytest.fixture
def documents_app_test(tmp_path: Path, monkeypatch) -> Iterator[AppTest]:
    """Create an AppTest instance for the Documents page with stubs for side effects."""
    from src.config.settings import settings as app_settings

    app_settings.data_dir = tmp_path

    prev_modules = {
        name: sys.modules.get(name)
        for name in ("src.ui.ingest_adapter", "src.utils.storage")
    }

    # Stub ingestion adapter and storage helpers to avoid heavy imports.
    _install_stub("src.ui.ingest_adapter", ingest_files=lambda *_, **__: {"count": 0})
    _install_stub(
        "src.utils.storage",
        create_vector_store=lambda *_, **__: object(),
        FALLBACK_SPARSE_MODEL="bm25",
        PREFERRED_SPARSE_MODEL="bm25",
        ensure_hybrid_collection=lambda *_, **__: None,
        get_client_config=lambda *_, **__: {},
    )

    # Import the page module and patch runtime helpers.
    page_mod = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(page_mod, "settings", app_settings, raising=False)
    monkeypatch.setattr(page_mod, "build_router_engine", lambda *_, **__: object())
    monkeypatch.setattr(page_mod, "export_graph_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(page_mod, "export_graph_parquet", lambda *_, **__: None)

    # Seed session state with minimal indices so the snapshot utilities render.
    class _DummyStorage:
        def persist(self, persist_dir: str) -> None:
            path = Path(persist_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "ok").write_text("1", encoding="utf-8")

    class _VecIndex:
        def __init__(self) -> None:
            self.storage_context = _DummyStorage()

    class _GraphStore:
        def persist(self, persist_dir: str) -> None:
            path = Path(persist_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "ok").write_text("1", encoding="utf-8")

        def get_nodes(self):
            yield from []

    class _PgIndex:
        def __init__(self) -> None:
            self.property_graph_store = _GraphStore()

    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "02_documents.py"
    at = AppTest.from_file(str(page_path))
    at.default_timeout = 6
    at.session_state["vector_index"] = _VecIndex()
    at.session_state["graphrag_index"] = _PgIndex()
    try:
        yield at
    finally:
        for name in ("src.ui.ingest_adapter", "src.utils.storage"):
            sys.modules.pop(name, None)
            original = prev_modules.get(name)
            if original is not None:
                sys.modules[name] = original


@pytest.mark.integration
def test_snapshot_rebuild_button(documents_app_test: AppTest, tmp_path: Path) -> None:
    """Click the rebuild button and assert the snapshot directory is created."""
    app = documents_app_test.run()
    assert not app.exception

    rebuild_buttons = [b for b in app.button if "Rebuild GraphRAG Snapshot" in str(b)]
    assert rebuild_buttons, "Rebuild button not found"

    result = rebuild_buttons[0].click().run()

    storage = tmp_path / "storage"
    found = False
    for _ in range(10):
        if storage.exists() and any(
            p.is_dir() and not p.name.startswith("_tmp-") for p in storage.iterdir()
        ):
            found = True
            break
        time.sleep(0.05)

    assert found, "final snapshot directory not created"
    success_messages = [msg.value for msg in result.success]
    assert any("Snapshot rebuilt" in value for value in success_messages)
