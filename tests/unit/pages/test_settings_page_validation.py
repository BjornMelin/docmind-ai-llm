from __future__ import annotations

import importlib
import sys
import types
from contextlib import nullcontext
from typing import Any, cast

import pytest


class _FailingState(dict[str, object]):
    def __init__(self, initial: dict[str, object], fail_key: str) -> None:
        super().__init__(initial)
        self._fail_key = fail_key
        self._armed = True

    def __setitem__(self, key: str, value: object) -> None:
        super().__setitem__(key, value)
        if self._armed and key == self._fail_key:
            self._armed = False
            raise RuntimeError(f"clear failed for {key}")

    def pop(self, key: str, *args: object) -> object:
        value = super().pop(key, *args)
        if self._armed and key == self._fail_key:
            self._armed = False
            raise RuntimeError(f"clear failed for {key}")
        return value


def _load_settings_page_module(
    monkeypatch: pytest.MonkeyPatch,
) -> types.ModuleType:
    # Avoid importing heavy UI runtime init for unit tests.
    integrations = importlib.import_module("src.config.integrations")
    monkeypatch.setattr(integrations, "initialize_integrations", lambda **_: None)

    # Ensure a fresh import for each test (avoid module cache pollution).
    sys.modules.pop("src.pages.04_settings", None)
    return importlib.import_module("src.pages.04_settings")


@pytest.mark.unit
def test_validate_candidate_formats_error_locations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)

    validated, errors = page._validate_candidate({"llm_backend": "nope"})

    assert validated is None
    assert errors
    assert any("llm_backend" in msg for msg in errors)


@pytest.mark.unit
def test_validate_candidate_handles_type_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)

    validated, errors = page._validate_candidate({"llm_backend": object()})

    assert validated is None
    assert errors


@pytest.mark.unit
def test_validate_candidate_rejects_blank_model_cache_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)

    validated, errors = page._validate_candidate(
        {"parsing": {"model_cache_dir": ""}},
    )

    assert validated is None
    assert any(
        "parsing.model_cache_dir" in error and "must not be empty" in error
        for error in errors
    )


@pytest.mark.unit
def test_render_actions_rechecks_ui_errors_before_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)
    applied: list[object] = []
    errors: list[str] = []

    monkeypatch.setattr(
        page.st, "columns", lambda _count: [nullcontext(), nullcontext()]
    )
    monkeypatch.setattr(
        page.st,
        "button",
        lambda label, **_kwargs: label == "Apply runtime",
    )
    monkeypatch.setattr(page.st, "error", lambda message: errors.append(str(message)))
    monkeypatch.setattr(page, "_apply_validated_runtime", applied.append)

    page._render_actions(page.settings, ["openai.default_headers: invalid JSON"])

    assert applied == []
    assert errors == ["Cannot apply: invalid settings."]


@pytest.mark.unit
def test_render_actions_keeps_save_enabled_during_background_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)
    disabled: dict[str, bool] = {}
    infos: list[str] = []

    class _Manager:
        def activity_snapshot(self) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                has_active_jobs=True,
                foreground_runtime_active=False,
                maintenance_active=False,
            )

    monkeypatch.setattr(page, "get_job_manager", lambda: _Manager())
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda _count: [nullcontext(), nullcontext()],
    )

    def _button(label: str, **kwargs: object) -> bool:
        disabled[label] = bool(kwargs.get("disabled"))
        return False

    monkeypatch.setattr(page.st, "button", _button)
    monkeypatch.setattr(page.st, "info", lambda message: infos.append(str(message)))

    page._render_actions(page.settings, [])

    assert disabled == {"Apply runtime": True, "Save": False}
    assert infos == [
        "Background work is active. You can save settings now, but apply them "
        "after the work finishes."
    ]


@pytest.mark.unit
def test_runtime_controls_are_disabled_during_foreground_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)
    disabled: dict[str, bool] = {}
    infos: list[str] = []

    class _Manager:
        def activity_snapshot(self) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                has_active_jobs=False,
                foreground_runtime_active=True,
                maintenance_active=False,
            )

    manager = _Manager()
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        page.st,
        "columns",
        lambda _count: [nullcontext(), nullcontext()],
    )

    def _button(label: str, **kwargs: object) -> bool:
        disabled[label] = bool(kwargs.get("disabled"))
        return False

    monkeypatch.setattr(page.st, "button", _button)
    monkeypatch.setattr(page.st, "info", lambda message: infos.append(str(message)))
    monkeypatch.setattr(page.st, "subheader", lambda _message: None)
    monkeypatch.setattr(page.st, "caption", lambda _message: None)

    page._render_actions(page.settings, [])
    page._render_cache_controls()

    assert disabled == {
        "Apply runtime": True,
        "Save": False,
        "Clear caches": True,
    }
    assert infos == [
        "A live runtime operation is active. Save remains available; apply after "
        "it finishes.",
        "Cache clearing is unavailable while the live runtime is in use.",
    ]


@pytest.mark.unit
def test_apply_runtime_rechecks_background_work_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)
    previous_settings = page.settings.model_copy(deep=True)
    warnings: list[str] = []

    class _Manager:
        def admission_quiescence(self) -> None:
            raise page.JobConflictError("active")

        def activity_snapshot(self) -> types.SimpleNamespace:
            return types.SimpleNamespace(foreground_runtime_active=False)

    monkeypatch.setattr(page, "get_job_manager", lambda: _Manager())
    monkeypatch.setattr(
        page,
        "_apply_validated_runtime_quiesced",
        lambda _validated: pytest.fail("runtime mutated without quiescence"),
    )
    monkeypatch.setattr(
        page.st,
        "warning",
        lambda message: warnings.append(str(message)),
    )

    page._apply_validated_runtime(page.settings.model_copy(deep=True))

    assert page.settings == previous_settings
    assert warnings == [
        "Runtime changes are unavailable while background work is active."
    ]


@pytest.mark.unit
def test_apply_runtime_failure_rolls_back_settings_and_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed provider bind cannot leave config or global LLM half-applied."""
    page = _load_settings_page_module(monkeypatch)
    integrations = importlib.import_module("src.config.integrations")
    from llama_index.core import Settings as LISettings

    previous_settings = page.settings.model_copy(deep=True)
    previous_llm = object()
    previous_embed_model = object()
    monkeypatch.setattr(LISettings, "_llm", previous_llm)
    monkeypatch.setattr(LISettings, "_embed_model", previous_embed_model)
    monkeypatch.setattr(
        LISettings,
        "_prompt_helper",
        types.SimpleNamespace(context_window=12345, num_output=321),
    )
    LISettings.context_window = 12345
    LISettings.num_output = 321

    def _fail(**_kwargs: object) -> None:
        mutable_settings = cast(Any, LISettings)
        mutable_settings._llm = object()
        mutable_settings._embed_model = object()
        LISettings.context_window = 54321
        LISettings.num_output = 123
        raise RuntimeError("bind failed")

    monkeypatch.setattr(
        integrations,
        "initialize_integrations",
        _fail,
    )
    errors: list[str] = []
    events: list[dict[str, object]] = []
    invalidations: list[None] = []
    chat_runtime = importlib.import_module("src.ui.chat_runtime")
    monkeypatch.setattr(
        chat_runtime,
        "invalidate_coordinator",
        lambda: invalidations.append(None),
    )
    monkeypatch.setattr(page.st, "error", lambda message: errors.append(str(message)))
    monkeypatch.setattr(page, "log_jsonl", lambda event: events.append(event))

    validated = page.DocMindSettings(
        llm_backend="lmstudio",
        llm_request={"model": "replacement/model"},
        _env_file=None,
    )
    page._apply_validated_runtime(validated)

    assert page.settings == previous_settings
    assert LISettings._llm is previous_llm
    assert LISettings._embed_model is previous_embed_model
    assert LISettings.context_window == 12345
    assert LISettings.num_output == 321
    assert errors == ["Runtime apply failed: RuntimeError"]
    assert events[-1]["success"] is False
    assert invalidations == []


@pytest.mark.unit
def test_apply_runtime_key_error_rolls_back_complete_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any ordinary exception at the UI boundary restores every runtime global."""
    page = _load_settings_page_module(monkeypatch)
    integrations = importlib.import_module("src.config.integrations")
    vector_session = importlib.import_module("src.ui.vector_session")
    chat_runtime = importlib.import_module("src.ui.chat_runtime")
    from llama_index.core import Settings as LISettings

    previous_settings = page.settings.model_copy(deep=True)
    previous_llm = object()
    previous_embed_model = object()
    monkeypatch.setattr(LISettings, "_llm", previous_llm)
    monkeypatch.setattr(LISettings, "_embed_model", previous_embed_model)
    monkeypatch.setattr(
        LISettings,
        "_prompt_helper",
        types.SimpleNamespace(context_window=12345, num_output=321),
    )
    LISettings.context_window = 12345
    LISettings.num_output = 321

    def _bind(**_kwargs: object) -> None:
        mutable_settings = cast(Any, LISettings)
        mutable_settings._llm = object()
        mutable_settings._embed_model = object()
        LISettings.context_window = 54321
        LISettings.num_output = 123

    def _fail_clear(*_args: object, **_kwargs: object) -> None:
        raise KeyError("unexpected clear failure")

    class _Manager:
        admission_quiescence = staticmethod(nullcontext)

    errors: list[str] = []
    events: list[dict[str, object]] = []
    monkeypatch.setattr(page, "get_job_manager", lambda: _Manager())
    monkeypatch.setattr(integrations, "initialize_integrations", _bind)
    monkeypatch.setattr(vector_session, "clear_session_runtime", _fail_clear)
    monkeypatch.setattr(
        chat_runtime,
        "invalidate_coordinator",
        lambda: pytest.fail("coordinator invalidated after failed apply"),
    )
    monkeypatch.setattr(page.st, "error", lambda message: errors.append(str(message)))
    monkeypatch.setattr(page, "log_jsonl", lambda event: events.append(event))
    validated = page.DocMindSettings(
        llm_backend="lmstudio",
        llm_request={"model": "replacement/model"},
        _env_file=None,
    )

    page._apply_validated_runtime(validated)

    assert page.settings == previous_settings
    assert page.settings.cache_version == previous_settings.cache_version
    assert LISettings._llm is previous_llm
    assert LISettings._embed_model is previous_embed_model
    assert LISettings.context_window == 12345
    assert LISettings.num_output == 321
    assert errors == ["Runtime apply failed: KeyError"]
    assert events[-1]["success"] is False
    assert events[-1]["error_type"] == "KeyError"


@pytest.mark.unit
def test_apply_runtime_success_invalidates_stale_chat_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful provider bind advances and invalidates the Chat runtime."""
    page = _load_settings_page_module(monkeypatch)
    integrations = importlib.import_module("src.config.integrations")
    chat_runtime = importlib.import_module("src.ui.chat_runtime")
    from llama_index.core import Settings as LISettings

    bound_llm = object()
    previous_version = page.settings.cache_version
    invalidations: list[None] = []
    lifecycle_events: list[str] = []

    class _Router:
        def close(self) -> None:
            lifecycle_events.append("router")

    def _bind(**_kwargs: object) -> None:
        monkeypatch.setattr(LISettings, "_llm", bound_llm)

    def _invalidate() -> None:
        lifecycle_events.append("coordinator")
        invalidations.append(None)

    monkeypatch.setattr(integrations, "initialize_integrations", _bind)
    monkeypatch.setattr(
        chat_runtime,
        "invalidate_coordinator",
        _invalidate,
    )
    monkeypatch.setattr(
        page.st,
        "session_state",
        {"router_engine": _Router()},
    )
    successes: list[str] = []
    monkeypatch.setattr(
        page.st, "success", lambda message: successes.append(str(message))
    )
    monkeypatch.setattr(page, "log_jsonl", lambda _event: None)

    validated = page.DocMindSettings(
        llm_backend="lmstudio",
        llm_request={"model": "replacement/model"},
        _env_file=None,
    )
    page._apply_validated_runtime(validated)

    assert LISettings._llm is bound_llm
    assert page.settings.cache_version == previous_version + 1
    assert invalidations == [None]
    assert lifecycle_events == ["router", "coordinator"]
    assert successes == ["Runtime applied. Settings.llm bound; Chat runtime refreshed."]


@pytest.mark.unit
@pytest.mark.parametrize(
    "fail_key",
    [
        "_vector_index_resource",
        "_vector_runtime_generation",
        "vector_index",
        "router_engine",
        "_router_runtime_generation",
        "graphrag_index",
        "_snapshot_collections",
        "_snapshot_loaded_id",
    ],
)
def test_apply_runtime_clear_failure_rolls_back_every_assignment(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
    fail_key: str,
) -> None:
    page = _load_settings_page_module(monkeypatch)
    integrations = importlib.import_module("src.config.integrations")
    chat_runtime = importlib.import_module("src.ui.chat_runtime")
    from llama_index.core import Settings as LISettings

    from src.ui.vector_session import (
        VectorIndexResource,
        replace_session_runtime,
        retire_session_runtime_resources,
    )

    class _Client:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class _Router:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    previous_settings = page.settings.model_copy(deep=True)
    previous_llm = object()
    previous_embed_model = object()
    monkeypatch.setattr(LISettings, "_llm", previous_llm)
    monkeypatch.setattr(LISettings, "_embed_model", previous_embed_model)
    monkeypatch.setattr(
        LISettings,
        "_prompt_helper",
        types.SimpleNamespace(context_window=12345, num_output=321),
    )
    LISettings.context_window = 12345
    LISettings.num_output = 321
    client = _Client()
    router = _Router()
    resource = VectorIndexResource("old-index", client=client)
    initial_state: dict[str, object] = {}
    replace_session_runtime(
        initial_state,
        resource,
        router,
        runtime_generation=previous_settings.cache_version,
        state_updates={
            "graphrag_index": "old-graph",
            "_snapshot_collections": {"text": "old-text"},
            "_snapshot_loaded_id": "old-snapshot",
        },
    )
    state = _FailingState(initial_state, fail_key)
    expected_state = dict(initial_state)
    errors: list[str] = []

    class _Manager:
        admission_quiescence = staticmethod(nullcontext)

    def _bind(**_kwargs: object) -> None:
        mutable_settings = cast(Any, LISettings)
        mutable_settings._llm = object()
        mutable_settings._embed_model = object()
        LISettings.context_window = 54321
        LISettings.num_output = 123

    monkeypatch.setattr(page, "get_job_manager", lambda: _Manager())
    monkeypatch.setattr(page.st, "session_state", state)
    monkeypatch.setattr(integrations, "initialize_integrations", _bind)
    monkeypatch.setattr(
        chat_runtime,
        "invalidate_coordinator",
        lambda: pytest.fail("coordinator invalidated after failed apply"),
    )
    monkeypatch.setattr(page.st, "error", lambda message: errors.append(str(message)))
    monkeypatch.setattr(page, "log_jsonl", lambda _event: None)
    validated = page.DocMindSettings(
        llm_backend="lmstudio",
        llm_request={"model": "replacement/model"},
        _env_file=None,
    )

    try:
        page._apply_validated_runtime(validated)

        assert page.settings == previous_settings
        assert page.settings.cache_version == previous_settings.cache_version
        assert LISettings._llm is previous_llm
        assert LISettings._embed_model is previous_embed_model
        assert LISettings.context_window == 12345
        assert LISettings.num_output == 321
        assert state == expected_state
        assert client.close_calls == 0
        assert router.close_calls == 0
        assert errors == ["Runtime apply failed: RuntimeError"]
    finally:
        retire_session_runtime_resources()


@pytest.mark.unit
def test_web_search_warning_matches_least_privilege_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)
    warnings: list[str] = []
    monkeypatch.setattr(
        page.st, "warning", lambda message: warnings.append(str(message))
    )

    page._render_ollama_web_search_warning(
        enabled=True,
        allow_remote=False,
        allowlist=["https://ollama.com"],
    )
    page._render_ollama_web_search_warning(
        enabled=True,
        allow_remote=True,
        allowlist=[],
    )

    assert warnings == []


@pytest.mark.unit
def test_web_search_warning_requests_exact_ollama_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)
    warnings: list[str] = []
    monkeypatch.setattr(
        page.st, "warning", lambda message: warnings.append(str(message))
    )

    page._render_ollama_web_search_warning(
        enabled=True,
        allow_remote=False,
        allowlist=["https://api.example.com"],
    )

    assert warnings == [
        "Ollama web tools require `https://ollama.com` in "
        "`DOCMIND_SECURITY__ENDPOINT_ALLOWLIST`."
    ]


@pytest.mark.unit
def test_validate_candidate_formats_nested_error_locations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)

    validated, errors = page._validate_candidate(
        {
            "llm_backend": "ollama",
            "security": {"allow_remote_endpoints": "nope"},
        }
    )

    assert validated is None
    assert errors
    assert any("security.allow_remote_endpoints" in msg for msg in errors)


@pytest.mark.unit
def test_validate_llamacpp_inputs_requires_server_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)

    errors = page._validate_llamacpp_inputs("llamacpp", "")

    assert errors == ["Provide a llama.cpp OpenAI-compatible server URL."]


@pytest.mark.unit
def test_validate_llamacpp_inputs_accepts_explicit_localhost_1234(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)

    errors = page._validate_llamacpp_inputs(
        "llamacpp",
        "http://localhost:1234/v1",
    )

    assert errors == []


@pytest.mark.unit
def test_validate_llamacpp_inputs_rejects_explicit_default_llamacpp_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)

    errors = page._validate_llamacpp_inputs(
        "llamacpp",
        "https://api.openai.com/v1",
    )

    assert errors == ["Provide a llama.cpp OpenAI-compatible server URL."]


@pytest.mark.unit
def test_build_endpoint_test_headers_includes_llamacpp_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)

    validated = page.DocMindSettings.model_validate(
        {
            "llm_backend": "llamacpp",
            "llamacpp_base_url": "http://localhost:8080/v1",
            "openai": {
                "api_key": "local-token",
                "default_headers": {"X-Provider": "llamacpp"},
            },
        }
    )

    headers = page._build_endpoint_test_headers(validated)

    assert headers == {
        "Accept": "application/json",
        "Authorization": "Bearer local-token",
        "X-Provider": "llamacpp",
    }
