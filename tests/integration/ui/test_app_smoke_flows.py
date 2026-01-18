"""Fast AppTest smoke + critical flow checks for Streamlit pages."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.mark.integration
def test_app_entrypoint_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the root app entrypoint renders without loading heavy pages."""
    captured: dict[str, object] = {}

    def _navigation(pages):
        captured["pages"] = pages

        class _Nav:
            def run(self) -> None:
                st.write("Navigation ready")

        return _Nav()

    monkeypatch.setattr(st, "navigation", _navigation)
    root = Path(__file__).resolve().parents[3]
    app = AppTest.from_file(str(root / "app.py")).run()

    assert not app.exception
    assert app.markdown

    pages = captured.get("pages") or []
    assert len(pages) == 4
    titles = [getattr(page, "title", None) for page in pages]
    assert [t for t in titles if t] == ["Chat", "Documents", "Analytics", "Settings"]


@pytest.fixture
def documents_app(tmp_path: Path) -> Generator[AppTest]:
    """AppTest harness for Documents page with temp data dir."""
    from src.config.settings import settings as app_settings

    original_data_dir = app_settings.data_dir
    app_settings.data_dir = tmp_path

    st.cache_resource.clear()
    st.cache_data.clear()

    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "02_documents.py"
    app = AppTest.from_file(str(page_path))
    app.default_timeout = 6

    try:
        yield app
    finally:
        app_settings.data_dir = original_data_dir
        st.cache_resource.clear()
        st.cache_data.clear()


@pytest.mark.integration
def test_documents_empty_ingest_warns(documents_app: AppTest) -> None:
    """Clicking ingest with no files should warn and not crash."""
    app = documents_app.run()
    assert not app.exception

    ingest_buttons = [b for b in app.button if b.label == "Ingest"]
    assert ingest_buttons, "Ingest button not found"

    result = ingest_buttons[0].click().run()
    assert not result.exception
    warnings = [str(getattr(w, "value", "")) for w in result.warning]
    assert any("No files selected" in msg for msg in warnings)


@pytest.fixture
def analytics_app(tmp_path: Path) -> Generator[AppTest]:
    """AppTest harness for Analytics page with analytics disabled."""
    from src.config.settings import settings as app_settings

    original_data_dir = app_settings.data_dir
    original_enabled = bool(getattr(app_settings, "analytics_enabled", False))

    app_settings.data_dir = tmp_path
    app_settings.analytics_enabled = False

    st.cache_resource.clear()
    st.cache_data.clear()

    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "03_analytics.py"
    app = AppTest.from_file(str(page_path))
    app.default_timeout = 6

    try:
        yield app
    finally:
        app_settings.data_dir = original_data_dir
        app_settings.analytics_enabled = original_enabled
        st.cache_resource.clear()
        st.cache_data.clear()


@pytest.mark.integration
def test_analytics_disabled_message(analytics_app: AppTest) -> None:
    """Analytics page should show a disabled notice when gated off."""
    app = analytics_app.run()
    assert not app.exception

    infos = [str(getattr(info, "value", "")) for info in app.info]
    assert any("Analytics disabled" in msg for msg in infos)
