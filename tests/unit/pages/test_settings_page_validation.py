from __future__ import annotations

import importlib
import sys
import types

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
def test_validate_llamacpp_inputs_accepts_openai_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)

    errors = page._validate_llamacpp_inputs(
        "llamacpp",
        "",
        "http://localhost:8080/v1",
    )

    assert errors == []


@pytest.mark.unit
def test_validate_llamacpp_inputs_rejects_default_openai_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)

    errors = page._validate_llamacpp_inputs(
        "llamacpp",
        "",
        "https://api.openai.com/v1",
    )

    assert errors == ["Provide a llama.cpp OpenAI-compatible server URL."]


@pytest.mark.unit
def test_validate_llamacpp_inputs_accepts_explicit_default_llamacpp_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _load_settings_page_module(monkeypatch)

    errors = page._validate_llamacpp_inputs(
        "llamacpp",
        "https://api.openai.com/v1",
    )

    assert errors == []


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
