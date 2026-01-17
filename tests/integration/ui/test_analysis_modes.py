"""Integration test validating analysis mode UI wiring (SPEC-036)."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import src.ui.background_jobs as bg

pytestmark = pytest.mark.integration


class _FakeQueryEngine:
    def __init__(self, doc_id: str) -> None:
        self._doc_id = doc_id

    def query(self, _query: str) -> object:
        node = SimpleNamespace(metadata={"doc_id": self._doc_id})
        src = SimpleNamespace(node=node)
        return SimpleNamespace(response=f"answer:{self._doc_id}", source_nodes=[src])


class _FakeVectorIndex:
    def as_query_engine(self, **kwargs: object) -> _FakeQueryEngine:
        doc_id = "combined"
        filters = kwargs.get("filters")
        parts = getattr(filters, "filters", None)
        if isinstance(parts, list) and parts:
            value = getattr(parts[0], "value", None)
            if value is not None:
                doc_id = str(value)
        return _FakeQueryEngine(doc_id)


@pytest.fixture
def chat_analysis_app_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    from src.config.settings import settings as app_settings

    original_data_dir = app_settings.data_dir
    original_chat_path = app_settings.chat.sqlite_path
    original_db_path = app_settings.database.sqlite_db_path

    app_settings.data_dir = tmp_path
    app_settings.chat.sqlite_path = tmp_path / "chat.db"
    app_settings.database.sqlite_db_path = tmp_path / "docmind.db"

    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    (uploads_dir / "a.txt").write_text("a", encoding="utf-8")
    (uploads_dir / "b.txt").write_text("b", encoding="utf-8")

    # Replace background job manager with a synchronous fake so results render
    # deterministically in the same AppTest run.
    class _State:
        def __init__(self, *, owner_id: str, status: str, result, error: str | None):
            self.owner_id = owner_id
            self.status = status
            self.result = result
            self.error = error

    class _FakeJobManager:
        def __init__(self) -> None:
            self._events: dict[str, list[bg.ProgressEvent]] = {}
            self._states: dict[str, _State] = {}

        def start_job(self, *, owner_id: str, fn):  # type: ignore[no-untyped-def]
            job_id = "job-1"
            events: list[bg.ProgressEvent] = []

            def _report(evt: bg.ProgressEvent) -> None:
                events.append(evt)

            try:
                res = fn(threading.Event(), _report)
                state = _State(
                    owner_id=owner_id, status="succeeded", result=res, error=None
                )
            except Exception as exc:  # pragma: no cover - defensive
                state = _State(
                    owner_id=owner_id, status="failed", result=None, error=str(exc)
                )
            self._events[job_id] = events
            self._states[job_id] = state
            return job_id

        def get(self, job_id: str, *, owner_id: str):  # type: ignore[no-untyped-def]
            state = self._states.get(job_id)
            if state is None or state.owner_id != owner_id:
                return None
            return state

        def drain_progress(self, job_id: str, *, owner_id: str, max_events: int = 100):  # type: ignore[no-untyped-def]
            _ = max_events
            state = self._states.get(job_id)
            if state is None or state.owner_id != owner_id:
                return []
            events = self._events.get(job_id, [])
            self._events[job_id] = []
            return events

        def cancel(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return True

    fake_mgr = _FakeJobManager()
    monkeypatch.setattr(bg, "get_job_manager", lambda *_a, **_k: fake_mgr)
    monkeypatch.setattr(bg, "get_or_create_owner_id", lambda: "owner")

    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "01_chat.py"
    at = AppTest.from_file(str(page_path))
    at.default_timeout = 6
    at.session_state["vector_index"] = _FakeVectorIndex()
    try:
        yield at
    finally:
        app_settings.data_dir = original_data_dir
        app_settings.chat.sqlite_path = original_chat_path
        app_settings.database.sqlite_db_path = original_db_path


def test_analysis_separate_mode_renders_per_doc_outputs(
    chat_analysis_app_test: AppTest,
) -> None:
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
