"""Fast AppTest smoke + critical flow checks for Streamlit pages."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
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

    history_error: Exception | None = None
    query_error: Exception | None = None
    response: AgentResponse | None = None

    def __init__(self, *_, **__) -> None:
        self._messages: list[object] = []

    def close(self) -> None:
        """Match the production coordinator lifecycle surface."""
        return None

    def process_query(
        self,
        *,
        query: str,
        settings_override: dict[str, object] | None = None,
        thread_id: str = "default",
        user_id: str = "local",
        checkpoint_id: str | None = None,
    ) -> AgentResponse:
        _ = (settings_override, thread_id, user_id, checkpoint_id)
        if self.query_error is not None:
            raise self.query_error
        return self.response or AgentResponse(content=f"Echo: {query}")

    def get_state_values(self, *_, **__) -> dict[str, object]:
        if self.history_error is not None:
            raise self.history_error
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


@pytest.mark.integration
def test_app_recovery_failure_stops_before_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistence recovery failures remain sanitized and fail closed."""
    from src.persistence import snapshot

    navigated: list[bool] = []
    monkeypatch.setattr(
        snapshot,
        "recover_snapshot_transactions",
        lambda _path: (_ for _ in ()).throw(RuntimeError("private recovery detail")),
    )
    monkeypatch.setattr(
        st,
        "navigation",
        lambda _pages: navigated.append(True),
    )
    st.cache_resource.clear()
    root = Path(__file__).resolve().parents[3]
    app = AppTest.from_file(str(root / "app.py"))
    app.default_timeout = apptest_timeout_sec()
    app = app.run()

    errors = [str(getattr(item, "value", "")) for item in app.error]
    assert errors == [
        "DocMind could not safely recover local persistence. "
        "Stop other writers and restart the app."
    ]
    assert "private recovery detail" not in str(app)
    assert navigated == []


@pytest.mark.integration
@pytest.mark.parametrize("artifact_mode", ["empty-cache", "incomplete-local"])
def test_chat_missing_artifacts_render_import_light_degraded_shell(
    tmp_path: Path,
    artifact_mode: str,
) -> None:
    """Real offline Chat renders degrade without model imports or network."""
    root = Path(__file__).resolve().parents[3]
    probe = textwrap.dedent(
        """
        import json
        import socket
        import sys
        from pathlib import Path

        from streamlit.testing.v1 import AppTest
        from src.config.settings import settings

        data_dir = Path(sys.argv[1])
        settings.data_dir = data_dir
        settings.chat.sqlite_path = data_dir / "chat.db"
        settings.embedding.cache_folder = data_dir / "empty-cache"
        settings.embedding.local_model_path = None
        settings.embedding.model_name = "docmind-tests/model-not-in-cache"
        settings.embedding.model_revision = "missing-revision"
        if sys.argv[3] == "incomplete-local":
            local_path = data_dir / "private-incomplete-model"
            local_path.mkdir(parents=True)
            (local_path / "config.json").write_text("{}", encoding="utf-8")
            settings.embedding.local_model_path = local_path
        settings.graphrag_cfg.autoload_policy = "ignore"
        network_calls = []

        def _deny_network(*args, **kwargs):
            network_calls.append((str(args), str(kwargs)))
            raise AssertionError("network access attempted")

        socket.socket.connect = _deny_network
        socket.create_connection = _deny_network
        page = Path(sys.argv[2]) / "src" / "pages" / "01_chat.py"
        app = AppTest.from_file(str(page), default_timeout=8).run()
        roots = {name.partition(".")[0] for name in sys.modules}
        payload = {
            "exceptions": [str(getattr(item, "value", "")) for item in app.exception],
            "warnings": [str(getattr(item, "value", "")) for item in app.warning],
            "captions": [str(getattr(item, "value", "")) for item in app.caption],
            "code": [str(getattr(item, "value", "")) for item in app.code],
            "network_calls": network_calls,
            "coordinator_loaded": "src.agents.coordinator" in sys.modules,
            "prohibited_roots": sorted(
                roots & {"torch", "transformers", "llama_index", "qdrant_client"}
            ),
        }
        print(json.dumps(payload))
        """
    )
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path), str(root), artifact_mode],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["exceptions"] == []
    assert payload["network_calls"] == []
    assert payload["coordinator_loaded"] is False
    assert payload["prohibited_roots"] == []
    expected_warning = (
        "Chat is unavailable because the configured local model is incomplete."
        if artifact_mode == "incomplete-local"
        else "Chat is unavailable because its local model artifacts are not installed."
    )
    assert payload["warnings"] == [expected_warning]
    assert "Sessions and local snapshot status remain available." in payload["captions"]
    if artifact_mode == "incomplete-local":
        assert str(tmp_path) not in str(payload)
        assert any(
            "remove DOCMIND_EMBEDDING__LOCAL_MODEL_PATH" in caption
            for caption in payload["captions"]
        )
        assert payload["code"] == []
    else:
        assert str(tmp_path) not in str(payload)
        assert payload["code"] == [
            "uv run python tools/models/pull.py --all --parser-defaults"
        ]


@pytest.fixture
def chat_app_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _reset_router_and_graph_modules: pytest.MonkeyPatch,
) -> Generator[AppTest, None, None]:
    """AppTest harness for Chat page with a stub coordinator.

    Args:
        tmp_path: Temporary directory for test data.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Generator yielding a configured AppTest instance.
    """
    from src.config.settings import settings as app_settings

    original_data_dir = app_settings.data_dir
    original_chat_path = app_settings.chat.sqlite_path
    original_autoload = app_settings.graphrag_cfg.autoload_policy

    app_settings.data_dir = tmp_path
    app_settings.chat.sqlite_path = tmp_path / "chat.db"
    app_settings.graphrag_cfg.autoload_policy = "ignore"
    _CoordinatorStub.history_error = None
    _CoordinatorStub.query_error = None
    _CoordinatorStub.response = None

    st.cache_resource.clear()
    st.cache_data.clear()

    mod = ModuleType("src.agents.coordinator")
    mod.MultiAgentCoordinator = _CoordinatorStub  # type: ignore[attr-defined]
    _reset_router_and_graph_modules.setitem(sys.modules, "src.agents.coordinator", mod)

    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "01_chat.py"
    app = AppTest.from_file(str(page_path))
    app.default_timeout = max(8, apptest_timeout_sec())

    try:
        yield app
    finally:
        app_settings.data_dir = original_data_dir
        app_settings.chat.sqlite_path = original_chat_path
        app_settings.graphrag_cfg.autoload_policy = original_autoload
        _CoordinatorStub.history_error = None
        _CoordinatorStub.query_error = None
        _CoordinatorStub.response = None
        st.cache_resource.clear()
        st.cache_data.clear()


@pytest.mark.integration
def test_chat_smoke_echo(chat_app_smoke: AppTest) -> None:
    """Chat page should accept input and render a stubbed response."""
    app = chat_app_smoke.run()
    assert not app.exception
    assert app.chat_input
    assert any(
        "No messages yet" in str(getattr(caption, "value", ""))
        for caption in app.caption
    )

    app = app.chat_input[0].set_value("hello").run()
    assert not app.exception

    user_texts = _chat_message_texts(app, avatar="user")
    assistant_texts = _chat_message_texts(app, avatar="assistant")
    assert any("hello" in text for text in user_texts)
    assert any("Echo: hello" in text for text in assistant_texts)


@pytest.mark.integration
def test_chat_corrupt_session_database_is_not_relabelled_as_missing_model(
    chat_app_smoke: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected persistence failures fail closed outside degraded-model UX."""
    from src.ui import chat_sessions

    monkeypatch.setattr(
        chat_sessions,
        "get_chat_db_conn",
        lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("database is malformed")),
    )

    app = chat_app_smoke.run()

    assert app.exception
    assert "database is malformed" in str(app.exception[0].value)
    assert not any(
        "local model artifacts" in str(getattr(item, "value", ""))
        for item in app.warning
    )
    assert not any(
        "tools/models/pull.py" in str(getattr(item, "value", "")) for item in app.code
    )


@pytest.mark.integration
def test_chat_history_failure_is_sanitized_and_retry_recovers(
    chat_app_smoke: AppTest,
) -> None:
    """A failed history read is explicit, sanitized, and normally retryable."""
    _CoordinatorStub.history_error = RuntimeError("history leaked-secret")

    app = chat_app_smoke.run()

    assert not app.exception
    assert not app.chat_input
    assert any(
        "Chat history could not be loaded" in str(getattr(error, "value", ""))
        for error in app.error
    )
    visible = " ".join(
        str(getattr(element, "value", ""))
        for collection in (app.error, app.caption)
        for element in collection
    )
    assert "leaked-secret" not in visible

    retry = next(
        button for button in app.button if button.label == "Retry chat history"
    )
    _CoordinatorStub.history_error = None
    app = retry.click().run()

    assert not app.exception
    assert app.chat_input
    assert not any(
        "Chat history could not be loaded" in str(getattr(error, "value", ""))
        for error in app.error
    )


@pytest.mark.integration
def test_chat_history_maintenance_blocks_prompt_and_offers_retry(
    chat_app_smoke: AppTest,
) -> None:
    """Maintenance cannot be mistaken for a new, writable conversation."""
    from src.ui.background_jobs import JobAdmissionPausedError

    _CoordinatorStub.history_error = JobAdmissionPausedError("maintenance")

    app = chat_app_smoke.run()

    assert not app.exception
    assert not app.chat_input
    assert any(
        "Chat history is unavailable during runtime maintenance"
        in str(getattr(info, "value", ""))
        for info in app.info
    )
    assert any(button.label == "Retry chat history" for button in app.button)


@pytest.mark.integration
def test_chat_provider_failure_preserves_sanitized_error_contract(
    chat_app_smoke: AppTest,
) -> None:
    """A synchronous provider failure renders the existing safe error contract."""
    app = chat_app_smoke.run()
    assert not app.exception
    _CoordinatorStub.query_error = RuntimeError("provider leaked-secret")

    app = app.chat_input[0].set_value("hello").run()

    assert not app.exception
    assert any(
        "Verify provider settings and endpoint connectivity"
        in str(getattr(error, "value", ""))
        for error in app.error
    )
    visible = " ".join(
        str(getattr(element, "value", ""))
        for collection in (app.error, app.caption)
        for element in collection
    )
    assert "Error id:" in visible
    assert "leaked-secret" not in visible


@pytest.mark.integration
def test_chat_completed_timeout_response_renders_as_terminal_markdown(
    chat_app_smoke: AppTest,
) -> None:
    """A completed timeout response is terminal content, not fake streaming."""
    _CoordinatorStub.response = AgentResponse(
        content="Request timed out.",
        sources=[],
        metadata={"reason": "timeout"},
        validation_score=0.0,
        processing_time=0.01,
        optimization_metrics={"timeout": True},
    )
    app = chat_app_smoke.run()

    app = app.chat_input[0].set_value("hello").run()

    assert not app.exception
    assistant_markdown = [
        str(item.value)
        for message in app.chat_message
        if getattr(message, "avatar", None) != "user"
        for item in getattr(message, "markdown", [])
    ]
    assert "Request timed out." in assistant_markdown


@pytest.fixture
def documents_app(tmp_path: Path) -> Generator[AppTest, None, None]:
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
) -> Generator[AppTest, None, None]:
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
        object.__setattr__(app_settings, "analytics_enabled", analytics_enabled)

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
def analytics_app(tmp_path: Path) -> Generator[AppTest, None, None]:
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
