"""Integration test validating analysis mode UI wiring (SPEC-036)."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from streamlit.testing.v1 import AppTest

import src.ui.background_jobs as bg
from src.analysis.models import AnalysisResult, PerDocResult
from src.ui.router_session import replace_session_router
from src.ui.vector_session import VectorIndexResource, replace_session_vector_resource
from tests.fixtures.vector_index import FakeVectorIndex
from tests.helpers.apptest_utils import apptest_timeout_sec

pytestmark = pytest.mark.integration


@pytest.fixture
def chat_analysis_app_test(  # noqa: PLR0915 - complete AppTest boundary fixture
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_job_manager,
    fake_job_owner_id,
) -> Iterator[AppTest]:
    """Provides a configured AppTest environment for validating analysis modes.

    This fixture prepares a temporary environment for UI testing by redirecting
    database paths to a temporary directory, creating mock upload files, and
    injecting a synchronous background job manager to ensure deterministic
    execution within the AppTest lifecycle.

    Args:
        tmp_path: The temporary directory path provided by pytest for file isolation.
        monkeypatch: The monkeypatch utility used to mock the background job system.
        fake_job_manager: A synchronous mock for the background job manager.
        fake_job_owner_id: A unique identifier for the simulated job owner.

    Returns:
        AppTest: A configured instance of the Streamlit AppTest for the chat page.
    """
    from src.config.settings import settings as app_settings

    original_data_dir = app_settings.data_dir
    original_chat_path = app_settings.chat.sqlite_path
    original_autoload = app_settings.graphrag_cfg.autoload_policy

    app_settings.data_dir = tmp_path
    app_settings.chat.sqlite_path = tmp_path / "chat.db"
    app_settings.graphrag_cfg.autoload_policy = "ignore"

    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    (uploads_dir / "a.txt").write_text("a", encoding="utf-8")
    (uploads_dir / "b.txt").write_text("b", encoding="utf-8")

    # Stub coordinator and chat sessions; analysis UI wiring does not depend on
    # coordinator graphs or chat DB state.
    coord_mod: Any = ModuleType("src.agents.coordinator")

    class _CoordStub:
        """Minimal coordinator stub for the analysis UI integration test."""

        def close(self) -> None:
            """Match the production coordinator lifecycle surface."""
            return None

        def list_checkpoints(self, *_args, **_kwargs) -> list[object]:
            """Return no checkpoints to keep UI rendering deterministic."""
            return []

    coord_mod.MultiAgentCoordinator = (  # type: ignore[attr-defined]
        lambda *_, **__: _CoordStub()
    )
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

    # Stub analysis service to keep UI test deterministic and fast under CI+cov.
    import src.analysis.service as analysis_service

    def _run_analysis_stub(*, documents, **_kwargs):  # type: ignore[no-untyped-def]
        per_doc = [
            PerDocResult(
                doc_id=str(getattr(d, "doc_id", "")),
                doc_name=str(getattr(d, "doc_name", "")),
                answer=f"answer: {getattr(d, 'doc_name', '')}",
                citations=[],
                duration_ms=0.0,
            )
            for d in list(documents or [])
        ]
        return AnalysisResult(
            mode="separate",
            per_doc=per_doc,
            combined=None,
            reduce=None,
            warnings=[],
            auto_decision_reason="stubbed",
        )

    monkeypatch.setattr(analysis_service, "run_analysis", _run_analysis_stub)

    # Replace background job manager with a synchronous fake so results render
    # deterministically in the same AppTest run.
    monkeypatch.setattr(bg, "get_job_manager", lambda *_a, **_k: fake_job_manager)
    monkeypatch.setattr(bg, "get_or_create_owner_id", lambda: fake_job_owner_id)

    # Analysis requires a vector resource bound to the active CURRENT snapshot.
    import src.persistence.snapshot as snapshot
    import src.persistence.snapshot_utils as snapshot_utils

    active_snapshot = tmp_path / "storage" / "active"
    active_snapshot.mkdir(parents=True)
    monkeypatch.setattr(
        snapshot, "latest_snapshot_dir", lambda *_a, **_k: active_snapshot
    )
    monkeypatch.setattr(snapshot, "load_manifest", lambda *_a, **_k: {"active": True})
    monkeypatch.setattr(snapshot_utils, "compute_staleness", lambda *_a, **_k: False)

    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "01_chat.py"
    at = AppTest.from_file(str(page_path), default_timeout=apptest_timeout_sec())
    vector_resource = VectorIndexResource(FakeVectorIndex())
    runtime_state: dict[str, object] = {}
    replace_session_vector_resource(
        runtime_state,
        vector_resource,
        runtime_generation=app_settings.cache_version,
    )
    replace_session_router(
        runtime_state,
        cast(Any, object()),
        runtime_generation=app_settings.cache_version,
    )
    for key, value in runtime_state.items():
        at.session_state[key] = value
    at.session_state["_snapshot_loaded_id"] = active_snapshot.name
    try:
        yield at
    finally:
        replace_session_vector_resource(
            runtime_state,
            None,
            runtime_generation=app_settings.cache_version,
        )
        app_settings.data_dir = original_data_dir
        app_settings.chat.sqlite_path = original_chat_path
        app_settings.graphrag_cfg.autoload_policy = original_autoload


def test_analysis_separate_mode_renders_per_doc_outputs(
    chat_analysis_app_test: AppTest,
) -> None:
    """Validates that the separate analysis mode renders unique outputs per document.

    Ensures that when 'separate' mode is selected, the UI correctly iterates
    through selected documents and displays the retrieved answer for each.

    Args:
        chat_analysis_app_test: The configured Streamlit AppTest helper fixture.

    Returns:
        None.
    """
    app = chat_analysis_app_test.run()
    assert not app.exception

    # Select separate mode and both uploaded docs.
    mode_select = next((sb for sb in app.selectbox if sb.key == "analysis_mode"), None)
    assert mode_select is not None
    app = mode_select.set_value("separate").run()
    assert not app.exception

    docs_multi = next((ms for ms in app.multiselect if ms.key == "analysis_docs"), None)
    assert docs_multi is not None
    selected_docs = list(getattr(docs_multi, "options", [])[:2])
    app = docs_multi.set_value(selected_docs).run()
    assert not app.exception

    query_area = next((ta for ta in app.text_area if ta.key == "analysis_query"), None)
    assert query_area is not None
    app = query_area.set_value("What is in these docs?").run()
    assert not app.exception

    run_button = next((btn for btn in app.button if btn.key == "analysis_run"), None)
    assert run_button is not None
    result = run_button.click().run()
    assert not result.exception

    subheaders = [sh.value for sh in result.subheader]
    assert any("Analysis (separate)" in str(v) for v in subheaders)
    assert any("answer:" in str(md.value) for md in result.markdown)
