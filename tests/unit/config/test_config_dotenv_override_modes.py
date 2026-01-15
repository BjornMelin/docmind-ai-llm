"""Tests for opt-in dotenv override and env masking/overlay."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _restore_env(keys: list[str], snapshot: dict[str, str]) -> None:
    for key in keys:
        if key in snapshot:
            os.environ[key] = snapshot[key]
        else:
            os.environ.pop(key, None)


def _reset_settings_module(mod) -> None:  # type: ignore[no-untyped-def]
    fresh = mod.DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    mod.apply_settings_in_place(mod.settings, fresh)
    mod._DOTENV_BOOTSTRAPPED = False  # type: ignore[attr-defined]
    mod._DOTENV_PRIORITY_MODE = None  # type: ignore[attr-defined]


def test_dotenv_first_overrides_env_except_security_and_overlays_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (repo / ".env").write_text(
        "\n".join(
            [
                "DOCMIND_LOG_LEVEL=DOTENV",
                "DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true",
                "DOCMIND_OPENAI__API_KEY=dotenv-key",
                "DOCMIND_CONFIG__DOTENV_PRIORITY=dotenv_first",
                "DOCMIND_CONFIG__ENV_MASK_KEYS=OPENAI_API_KEY",
                "DOCMIND_CONFIG__ENV_OVERLAY=OPENAI_API_KEY:openai.api_key",
                "",
            ]
        ),
        encoding="utf-8",
    )

    workdir = repo / "subdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    keys = [
        "DOCMIND_LOG_LEVEL",
        "DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS",
        "DOCMIND_CONFIG__DOTENV_PRIORITY",
        "DOCMIND_CONFIG__ENV_MASK_KEYS",
        "DOCMIND_CONFIG__ENV_OVERLAY",
        "OPENAI_API_KEY",
    ]
    snapshot = dict(os.environ)

    os.environ["DOCMIND_LOG_LEVEL"] = "ENV"
    os.environ["DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS"] = "false"
    os.environ["OPENAI_API_KEY"] = "global-key"

    import importlib

    mod = importlib.import_module("src.config.settings")
    try:
        mod.bootstrap_settings(force=True)
        assert mod.settings.log_level == "DOTENV"
        assert mod.settings.security.allow_remote_endpoints is False
        assert os.environ.get("OPENAI_API_KEY") == "dotenv-key"
    finally:
        _reset_settings_module(mod)
        _restore_env(keys, snapshot)


def test_env_first_keeps_env_over_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (repo / ".env").write_text(
        "\n".join(
            [
                "DOCMIND_LOG_LEVEL=DOTENV",
                "DOCMIND_CONFIG__DOTENV_PRIORITY=env_first",
                "",
            ]
        ),
        encoding="utf-8",
    )
    workdir = repo / "subdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    keys = ["DOCMIND_LOG_LEVEL", "DOCMIND_CONFIG__DOTENV_PRIORITY"]
    snapshot = dict(os.environ)
    os.environ["DOCMIND_LOG_LEVEL"] = "ENV"

    import importlib

    mod = importlib.import_module("src.config.settings")
    try:
        mod.bootstrap_settings(force=True)
        assert mod.settings.log_level == "ENV"
    finally:
        _reset_settings_module(mod)
        _restore_env(keys, snapshot)


def test_invalid_overlay_path_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (repo / ".env").write_text(
        "\n".join(
            [
                "DOCMIND_OPENAI__API_KEY=dotenv-key",
                "DOCMIND_CONFIG__ENV_OVERLAY=OPENAI_API_KEY:nope.path",
                "",
            ]
        ),
        encoding="utf-8",
    )
    workdir = repo / "subdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    snapshot = dict(os.environ)
    import importlib

    mod = importlib.import_module("src.config.settings")
    try:
        with pytest.raises(ValueError, match=r"Unknown settings path"):
            mod.bootstrap_settings(force=True)
    finally:
        _reset_settings_module(mod)
        _restore_env(["OPENAI_API_KEY", "DOCMIND_CONFIG__ENV_OVERLAY"], snapshot)
