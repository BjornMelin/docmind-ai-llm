"""Integration test validating the Documents page snapshot rebuild flow."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
from streamlit.testing.v1 import AppTest

from tests.helpers.apptest_utils import apptest_timeout_sec


def _install_stub(module_name: str, **attrs) -> None:
    """Register a lightweight stub module with the supplied attributes."""
    stub = ModuleType(module_name)
    for key, value in attrs.items():
        setattr(stub, key, value)
    sys.modules[module_name] = stub


@dataclass(slots=True)
class DocumentsAppHarness:
    """Test harness bundling the AppTest instance with stub state.

    Attributes:
        app: AppTest instance for the Documents page harness.
        rebuild_calls: List of snapshot rebuild paths requested by the UI.
    """

    app: AppTest
    rebuild_calls: list[Path]


@pytest.fixture
def documents_app_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[DocumentsAppHarness]:
    """Create an AppTest instance for the Documents page with stubs for side effects.

    Args:
        tmp_path: Temporary directory for test data.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Generator yielding a DocumentsAppHarness instance.
    """
    from src.config.settings import settings as app_settings

    monkeypatch.setenv("DOCMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DOCMIND_CACHE_DIR", str(tmp_path / "cache"))
    app_settings.data_dir = tmp_path
    app_settings.chat.sqlite_path = tmp_path / "chat.db"
    app_settings.database.sqlite_db_path = tmp_path / "docmind.db"

    prev_modules = {
        name: sys.modules.get(name)
        for name in ("src.ui.ingest_adapter", "src.utils.storage")
    }

    storage_base = tmp_path / "storage"

    # Stub ingestion adapter and storage helpers to avoid heavy imports.
    def _empty_ingest_result() -> dict[str, object]:
        return {
            "count": 0,
            "vector_index": None,
            "pg_index": None,
            "manifest": None,
            "exports": [],
            "duration_ms": 0.0,
            "metadata": {},
            "nlp_preview": None,
            "documents": [],
        }

    def _save_uploaded_file(*_, **__) -> tuple[Path, str]:
        dummy = tmp_path / "dummy"
        dummy.write_text("", encoding="utf-8")
        return dummy, "0"

    _install_stub(
        "src.ui.ingest_adapter",
        ingest_files=lambda *_, **__: _empty_ingest_result(),
        ingest_inputs=lambda *_, **__: _empty_ingest_result(),
        save_uploaded_file=_save_uploaded_file,
    )
    _install_stub(
        "src.utils.storage",
        create_vector_store=lambda *_, **__: object(),
        FALLBACK_SPARSE_MODEL="bm25",
        PREFERRED_SPARSE_MODEL="bm25",
        ensure_hybrid_collection=lambda *_, **__: None,
        get_client_config=lambda *_, **__: {},
    )

    # Stub snapshot rebuild boundary (UI wiring test).
    import src.persistence.snapshot as snapshot_mod
    import src.persistence.snapshot_service as snapshot_service

    rebuild_calls: list[Path] = []

    def _rebuild_snapshot(*_a, **_k) -> Path:
        snapshot_dir = storage_base / "snapshot-rebuild-test"
        assert snapshot_dir not in rebuild_calls, (
            "Rebuild should run once per test action."
        )
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        rebuild_calls.append(snapshot_dir)
        return snapshot_dir

    monkeypatch.setattr(
        snapshot_service,
        "rebuild_snapshot",
        _rebuild_snapshot,
    )
    monkeypatch.setattr(
        snapshot_mod,
        "load_manifest",
        lambda *_a, **_k: {
            "corpus_hash": "c" * 64,
            "config_hash": "f" * 64,
            "versions": {},
        },
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
    at = AppTest.from_file(str(page_path), default_timeout=apptest_timeout_sec())
    at.session_state["vector_index"] = _VecIndex()
    at.session_state["graphrag_index"] = _PgIndex()
    try:
        yield DocumentsAppHarness(app=at, rebuild_calls=rebuild_calls)
    finally:
        for name in ("src.ui.ingest_adapter", "src.utils.storage"):
            sys.modules.pop(name, None)
            original = prev_modules.get(name)
            if original is not None:
                sys.modules[name] = original


@pytest.mark.integration
def test_snapshot_rebuild_button(documents_app_test: DocumentsAppHarness) -> None:
    """Click the rebuild button and assert the snapshot directory is created."""
    app = documents_app_test.app.run()
    assert not app.exception

    rebuild_buttons = [b for b in app.button if "Rebuild GraphRAG Snapshot" in str(b)]
    assert rebuild_buttons, "Rebuild button not found"

    from src.persistence.snapshot import latest_snapshot_dir

    assert documents_app_test.rebuild_calls == []
    assert latest_snapshot_dir() is None

    result = rebuild_buttons[0].click().run()
    assert not result.exception

    assert len(documents_app_test.rebuild_calls) == 1
    snap = latest_snapshot_dir()
    assert snap is not None
    assert snap.is_dir()
    assert snap == documents_app_test.rebuild_calls[0]
    success_messages = [msg.value for msg in result.success]
    assert any("Snapshot rebuilt" in value for value in success_messages)
