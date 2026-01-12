"""Integration test: Chat autoloads latest non-stale snapshot into session."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from src.persistence.snapshot import (
    SnapshotManager,
    compute_config_hash,
    compute_corpus_hash,
)


@pytest.fixture
def chat_app_autoload(tmp_path: Path, monkeypatch) -> AppTest:
    # Point data_dir to tmp
    from src.config.settings import settings as _settings  # local import

    _settings.data_dir = tmp_path
    _settings.chat.sqlite_path = tmp_path / "chat.db"
    _settings.database.sqlite_db_path = tmp_path / "docmind.db"
    _settings.graphrag_cfg.autoload_policy = "latest_non_stale"

    # Create uploads dir with one file for stable corpus hash
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "a.txt").write_text("x", encoding="utf-8")

    # Write a snapshot using stubs
    mgr = SnapshotManager(tmp_path / "storage")
    tmp = mgr.begin_snapshot()

    class _Vec:
        class _S:
            def persist(self, persist_dir: str) -> None:
                p = Path(persist_dir)
                p.mkdir(parents=True, exist_ok=True)
                (p / "ok").write_text("1", encoding="utf-8")

        storage_context = _S()

    class _Store:
        def persist(self, persist_dir: str) -> None:
            p = Path(persist_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "ok").write_text("1", encoding="utf-8")

    mgr.persist_vector_index(_Vec(), tmp)
    mgr.persist_graph_store(_Store(), tmp)

    chash = compute_corpus_hash(list(uploads.glob("**/*")))
    cfg_hash = compute_config_hash({
        "router": _settings.retrieval.router,
        "hybrid": _settings.retrieval.enable_server_hybrid,
        "graph_enabled": _settings.enable_graphrag,
        "chunk_size": _settings.processing.chunk_size,
        "chunk_overlap": _settings.processing.chunk_overlap,
    })
    mgr.write_manifest(
        tmp,
        index_id="x",
        graph_store_type="property_graph",
        vector_store_type=_settings.database.vector_store_type,
        corpus_hash=chash,
        config_hash=cfg_hash,
        versions={"app": _settings.app_version},
    )
    final = mgr.finalize_snapshot(tmp)
    assert final.exists()

    # Stub LI modules for loader internals
    core_mod = ModuleType("llama_index.core")
    graph_mod = ModuleType("llama_index.core.graph_stores")

    class _StorageContext:
        def __init__(self, persist_dir: str) -> None:
            self.persist_dir = persist_dir

        @classmethod
        def from_defaults(cls, persist_dir: str):  # type: ignore[override]
            return cls(persist_dir)

    def _load_index_from_storage(storage: _StorageContext):  # type: ignore[override]
        # Validate vector sentinel
        assert (Path(storage.persist_dir) / "ok").exists()
        return SimpleNamespace(storage_dir=storage.persist_dir)

    class _PropertyGraphIndex:
        @staticmethod
        def from_existing(property_graph_store):  # type: ignore[override]
            return SimpleNamespace(property_graph_store=property_graph_store)

    class _SimplePropertyGraphStore:
        @staticmethod
        def from_persist_dir(persist_dir: str):  # type: ignore[override]
            assert (Path(persist_dir) / "ok").exists()
            return SimpleNamespace(persist_dir=persist_dir)

    core_mod.StorageContext = _StorageContext
    core_mod.load_index_from_storage = _load_index_from_storage
    core_mod.PropertyGraphIndex = _PropertyGraphIndex
    graph_mod.SimplePropertyGraphStore = _SimplePropertyGraphStore

    monkeypatch.setitem(sys.modules, "llama_index.core", core_mod)
    monkeypatch.setitem(sys.modules, "llama_index.core.graph_stores", graph_mod)

    # Stub router factory
    rf_mod = ModuleType("src.retrieval.router_factory")
    rf_mod.build_router_engine = lambda *_, **__: object()
    monkeypatch.setitem(sys.modules, "src.retrieval.router_factory", rf_mod)

    # Build AppTest for Chat page
    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "01_chat.py"
    at = AppTest.from_file(str(page_path))
    at.default_timeout = 6
    return at


@pytest.mark.integration
def test_chat_autoloads_router(chat_app_autoload: AppTest) -> None:
    app = chat_app_autoload.run()
    assert not app.exception
    # Router should be present in session_state
    assert "router_engine" in app.session_state
