"""Fast AppTest smoke + critical flow checks for Streamlit pages."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.mark.integration
def test_app_entrypoint_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the root app entrypoint renders without loading heavy pages.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    captured: dict[str, list[object]] = {}

    def _navigation(pages: list[object]) -> object:
        captured["pages"] = list(pages)

        class _Nav:
            def run(self) -> None:
                st.write("Navigation ready")

        return _Nav()

    monkeypatch.setattr(st, "navigation", _navigation)
    root = Path(__file__).resolve().parents[3]
    app = AppTest.from_file(str(root / "app.py"))
    app.default_timeout = 6
    app = app.run()

    assert not app.exception
    assert app.markdown

    pages = captured.get("pages", [])
    assert len(pages) == 4
    titles = [getattr(page, "title", None) for page in pages]
    assert [t for t in titles if t] == ["Chat", "Documents", "Analytics", "Settings"]


@pytest.fixture
def documents_app(tmp_path: Path) -> Generator[AppTest, None, None]:  # noqa: UP043
    """AppTest harness for Documents page with temp data dir.

    Args:
        tmp_path: Temporary directory for test data.

    Returns:
        Generator yielding a configured AppTest instance.
    """
    yield from _build_page_app(tmp_path, page_name="02_documents.py")


def _build_page_app(
    tmp_path: Path,
    *,
    page_name: str,
    analytics_enabled: bool | None = None,
) -> Generator[AppTest, None, None]:  # noqa: UP043
    """Create an AppTest instance with a temp data dir and optional flags.

    Args:
        tmp_path: Temporary directory for test data.
        page_name: Page file name under src/pages.
        analytics_enabled: Optional analytics flag override.

    Returns:
        Generator yielding a configured AppTest instance.
    """
    from src.config.settings import settings as app_settings

    original_data_dir = app_settings.data_dir
    original_enabled: bool | None = None

    app_settings.data_dir = tmp_path
    if analytics_enabled is not None:
        original_enabled = getattr(app_settings, "analytics_enabled", None)
        app_settings.analytics_enabled = analytics_enabled

    st.cache_resource.clear()
    st.cache_data.clear()

    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / page_name
    app = AppTest.from_file(str(page_path))
    app.default_timeout = 6

    try:
        yield app
    finally:
        app_settings.data_dir = original_data_dir
        if analytics_enabled is not None:
            object.__setattr__(app_settings, "analytics_enabled", original_enabled)
        st.cache_resource.clear()
        st.cache_data.clear()


@pytest.mark.integration
def test_documents_empty_ingest_warns(documents_app: AppTest) -> None:
    """Clicking ingest with no files should warn and not crash.

    Args:
        documents_app: Prepared AppTest fixture.

    Returns:
        None.
    """
    app = documents_app.run()
    assert not app.exception

    ingest_buttons = [b for b in app.button if b.label == "Ingest"]
    assert ingest_buttons, "Ingest button not found"

    result = ingest_buttons[0].click().run()
    assert not result.exception
    warnings = [str(getattr(w, "value", "")) for w in result.warning]
    assert any("No files selected" in msg for msg in warnings)


@pytest.fixture
def analytics_app(tmp_path: Path) -> Generator[AppTest, None, None]:  # noqa: UP043
    """AppTest harness for Analytics page with analytics disabled.

    Args:
        tmp_path: Temporary directory for test data.

    Returns:
        Generator yielding a configured AppTest instance.
    """
    yield from _build_page_app(
        tmp_path,
        page_name="03_analytics.py",
        analytics_enabled=False,
    )


@pytest.mark.integration
def test_analytics_disabled_message(analytics_app: AppTest) -> None:
    """Analytics page should show a disabled notice when gated off.

    Args:
        analytics_app: Prepared AppTest fixture.

    Returns:
        None.
    """
    app = analytics_app.run()
    assert not app.exception

    infos = [str(getattr(info, "value", "")) for info in app.info]
    assert any("Analytics disabled" in msg for msg in infos)
