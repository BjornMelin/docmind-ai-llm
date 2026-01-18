"""Fast AppTest smoke + critical flow checks for Streamlit pages."""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.agents.models import AgentResponse
from tests.helpers.apptest_utils import apptest_timeout_sec


class _CoordinatorStub:
    """Minimal coordinator stub for chat smoke tests."""

    def __init__(self, *_, **__) -> None:
        self._messages: list[object] = []

    def process_query(
        self,
        *,
        query: str,
        context: object | None = None,
        settings_override: dict[str, object] | None = None,
        thread_id: str = "default",
        user_id: str = "local",
        checkpoint_id: str | None = None,
    ) -> AgentResponse:
        _ = (context, settings_override, thread_id, user_id, checkpoint_id)
        return AgentResponse(content=f"Echo: {query}")

    def get_state_values(self, *_, **__) -> dict[str, object]:
        return {"messages": self._messages}

    def list_checkpoints(self, *_, **__) -> list[dict[str, object]]:
        return []


def _chat_message_texts(app: AppTest, *, avatar: str) -> list[str]:
    texts: list[str] = []
    for msg in app.chat_message:
        if getattr(msg, "avatar", None) != avatar:
            continue
        for attr in ("markdown", "text", "write"):
            items = getattr(msg, attr, [])
            if items:
                texts.append(str(items[0].value))
    return texts


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
    app.default_timeout = apptest_timeout_sec()
    app = app.run()

    assert not app.exception
    assert app.markdown

    pages = captured.get("pages", [])
    assert len(pages) == 4
    titles = [getattr(page, "title", None) for page in pages]
    assert [t for t in titles if t] == ["Chat", "Documents", "Analytics", "Settings"]


@pytest.fixture
def chat_app_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[AppTest, None, None]:  # noqa: UP043
    """AppTest harness for Chat page with a stub coordinator."""
    from src.config.settings import settings as app_settings

    original_data_dir = app_settings.data_dir
    original_chat_path = app_settings.chat.sqlite_path
    original_ops_path = app_settings.database.sqlite_db_path
    original_autoload = app_settings.graphrag_cfg.autoload_policy

    app_settings.data_dir = tmp_path
    app_settings.chat.sqlite_path = tmp_path / "chat.db"
    app_settings.database.sqlite_db_path = tmp_path / "docmind.db"
    app_settings.graphrag_cfg.autoload_policy = "ignore"

    st.cache_resource.clear()
    st.cache_data.clear()

    mod = ModuleType("src.agents.coordinator")
    mod.MultiAgentCoordinator = _CoordinatorStub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.agents.coordinator", mod)

    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "01_chat.py"
    app = AppTest.from_file(str(page_path))
    app.default_timeout = max(8, apptest_timeout_sec())

    try:
        yield app
    finally:
        app_settings.data_dir = original_data_dir
        app_settings.chat.sqlite_path = original_chat_path
        app_settings.database.sqlite_db_path = original_ops_path
        app_settings.graphrag_cfg.autoload_policy = original_autoload
        st.cache_resource.clear()
        st.cache_data.clear()


@pytest.mark.integration
def test_chat_smoke_echo(chat_app_smoke: AppTest) -> None:
    """Chat page should accept input and render a stubbed response."""
    app = chat_app_smoke.run()
    assert not app.exception
    assert app.chat_input

    app = app.chat_input[0].set_value("hello").run()
    assert not app.exception

    user_texts = _chat_message_texts(app, avatar="user")
    assistant_texts = _chat_message_texts(app, avatar="assistant")
    assert any("hello" in text for text in user_texts)
    assert any("Echo: hello" in text for text in assistant_texts)


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
    app.default_timeout = apptest_timeout_sec()

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
