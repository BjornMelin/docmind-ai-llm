from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest


def _load_settings_page_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    # Avoid importing heavy UI runtime init for unit tests.
    integrations = importlib.import_module("src.config.integrations")
    monkeypatch.setattr(integrations, "initialize_integrations", lambda **_: None)

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
def test_is_valid_gguf_path_rejects_outside_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _load_settings_page_module(monkeypatch)

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(page.Path, "home", lambda: fake_home)

    external = tmp_path / "external.gguf"
    external.write_text("dummy", encoding="utf-8")

    assert page._is_valid_gguf_path(str(external)) is False


@pytest.mark.unit
def test_is_valid_gguf_path_accepts_outside_home_when_base_dirs_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _load_settings_page_module(monkeypatch)

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(page.Path, "home", lambda: fake_home)

    allowed = tmp_path / "models"
    allowed.mkdir()
    gguf = allowed / "external.gguf"
    gguf.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        page.st,
        "session_state",
        {"docmind_allowed_gguf_base_dirs": [str(allowed)]},
        raising=False,
    )

    assert page._is_valid_gguf_path(str(gguf)) is True


@pytest.mark.unit
def test_is_valid_gguf_path_accepts_under_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _load_settings_page_module(monkeypatch)

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(page.Path, "home", lambda: fake_home)

    gguf = fake_home / "model.gguf"
    gguf.write_text("dummy", encoding="utf-8")

    assert page._is_valid_gguf_path(str(gguf)) is True
