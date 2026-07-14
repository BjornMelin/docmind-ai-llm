"""Integration test: Chat autoloads latest non-stale snapshot into session."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimplePropertyGraphStore
from streamlit.testing.v1 import AppTest

from src.persistence.snapshot import (
    SnapshotManager,
    compute_config_hash,
    compute_corpus_hash,
)
from src.persistence.snapshot_utils import current_config_dict
from tests.helpers.apptest_utils import apptest_timeout_sec


@pytest.fixture
def chat_app_autoload(tmp_path: Path, monkeypatch) -> AppTest:
    # Point data_dir to tmp
    from src.config.settings import settings as _settings  # local import

    _settings.data_dir = tmp_path
    _settings.chat.sqlite_path = tmp_path / "chat.db"
    _settings.graphrag_cfg.autoload_policy = "latest_non_stale"

    # Create uploads dir with one file for stable corpus hash
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "a.txt").write_text("x", encoding="utf-8")

    # Write a snapshot using stubs
    mgr = SnapshotManager(tmp_path / "storage")
    tmp = mgr.begin_snapshot()

    mgr.persist_graph_storage_context(
        StorageContext.from_defaults(property_graph_store=SimplePropertyGraphStore()),
        tmp,
    )

    chash = compute_corpus_hash(list(uploads.glob("**/*")), base_dir=uploads)
    cfg_hash = compute_config_hash(current_config_dict(_settings))
    mgr.write_manifest(
        tmp,
        index_id="x",
        graph_store_type="property_graph",
        vector_store_type=_settings.database.vector_store_type,
        text_collection="physical-text-v2",
        image_collection="physical-image-v2",
        corpus_hash=chash,
        config_hash=cfg_hash,
        versions={"app": _settings.app_version},
    )
    final = mgr.finalize_snapshot(tmp)
    assert final.exists()

    # Stub the coordinator and chat session helpers; this test only validates
    # snapshot hydration into session_state, not chat DB/coordinator behavior.
    class _CoordinatorStub:
        def __init__(self, *_, **__) -> None:
            return

        def close(self) -> None:
            return None

        def list_checkpoints(self, *_, **__) -> list[dict[str, object]]:
            return []

        def get_state_values(self, *_, **__) -> dict[str, object]:
            return {"messages": []}

        def process_query(self, *_, **__) -> object:
            return SimpleNamespace(content="ok", sources=[])

    coord_mod: Any = ModuleType("src.agents.coordinator")
    coord_mod.MultiAgentCoordinator = _CoordinatorStub
    monkeypatch.setitem(sys.modules, "src.agents.coordinator", coord_mod)

    @dataclass(frozen=True, slots=True)
    class _ChatSelection:
        thread_id: str
        user_id: str

    chat_sessions_mod: Any = ModuleType("src.ui.chat_sessions")
    chat_sessions_mod.ChatSelection = _ChatSelection
    chat_sessions_mod.get_chat_db_conn = lambda: object()
    chat_sessions_mod.render_session_sidebar = lambda _conn, **_kwargs: _ChatSelection(
        thread_id="t", user_id="local"
    )
    chat_sessions_mod.render_time_travel_sidebar = lambda *_, **__: None
    monkeypatch.setitem(sys.modules, "src.ui.chat_sessions", chat_sessions_mod)

    # Stub LI modules for loader internals
    core_mod: Any = ModuleType("llama_index.core")

    vector_store = SimpleNamespace(client=SimpleNamespace(close=lambda: None))

    class _VectorStoreIndex:
        @staticmethod
        def from_vector_store(store: object) -> SimpleNamespace:
            assert store is vector_store
            return SimpleNamespace(vector_store=store)

    class _PropertyGraphIndex:
        @staticmethod
        def from_existing(  # type: ignore[no-untyped-def]
            *, property_graph_store, vector_store, storage_context
        ):
            assert vector_store is storage_context.vector_store
            return SimpleNamespace(property_graph_store=property_graph_store)

    class _StorageContext:
        @staticmethod
        def from_defaults(*, persist_dir: str):  # type: ignore[no-untyped-def]
            graph_dir = Path(persist_dir)
            assert (graph_dir / "property_graph_store.json").exists()
            return SimpleNamespace(
                property_graph_store=SimpleNamespace(persist_dir=persist_dir),
                vector_store=SimpleNamespace(persist_dir=persist_dir),
            )

    core_mod.VectorStoreIndex = _VectorStoreIndex
    core_mod.PropertyGraphIndex = _PropertyGraphIndex
    core_mod.StorageContext = _StorageContext

    monkeypatch.setitem(sys.modules, "llama_index.core", core_mod)
    monkeypatch.setattr(
        "src.utils.storage.connect_vector_store",
        lambda *_args, **_kwargs: vector_store,
    )

    # Stub router factory
    rf_mod: Any = ModuleType("src.retrieval.router_factory")
    rf_mod.build_router_engine = lambda *_, **__: object()
    monkeypatch.setitem(sys.modules, "src.retrieval.router_factory", rf_mod)

    # Build AppTest for Chat page
    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "01_chat.py"
    return AppTest.from_file(str(page_path), default_timeout=apptest_timeout_sec())


@pytest.mark.integration
def test_chat_autoloads_router(chat_app_autoload: AppTest) -> None:
    app = chat_app_autoload.run()
    assert not app.exception
    # Router should be present in session_state
    assert "router_engine" in app.session_state
