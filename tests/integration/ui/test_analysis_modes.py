"""Integration test validating analysis mode UI wiring (SPEC-036)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.ui.background_jobs as bg
from tests.fixtures.vector_index import _FakeVectorIndex

pytestmark = pytest.mark.integration


@pytest.fixture
def chat_analysis_app_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_job_manager,
    fake_job_owner_id,
) -> AppTest:
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
    monkeypatch.setattr(bg, "get_job_manager", lambda *_a, **_k: fake_job_manager)
    monkeypatch.setattr(bg, "get_or_create_owner_id", lambda: fake_job_owner_id)

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
