from __future__ import annotations

import importlib
import sys
import types
from contextlib import nullcontext

import pytest


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
def test_apply_runtime_failure_rolls_back_settings_and_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed provider bind cannot leave config or global LLM half-applied."""
    page = _load_settings_page_module(monkeypatch)
    integrations = importlib.import_module("src.config.integrations")
    from llama_index.core import Settings as LISettings

    previous_settings = page.settings.model_copy(deep=True)
    previous_llm = object()
    monkeypatch.setattr(LISettings, "_llm", previous_llm)

    def _fail(**_kwargs: object) -> None:
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
    assert errors == ["Runtime apply failed: RuntimeError"]
    assert events[-1]["success"] is False
    assert invalidations == []


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
