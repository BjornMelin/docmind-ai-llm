"""Tests for explicit dotenv bootstrap and repo-root `.env` resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_resolve_dotenv_path_prefers_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (repo_root / ".env").write_text("DOCMIND_LOG_LEVEL=DOTENV\n", encoding="utf-8")

    subdir = repo_root / "subdir" / "nested"
    subdir.mkdir(parents=True)
    monkeypatch.chdir(subdir)

    from src.config.dotenv import resolve_dotenv_path

    assert resolve_dotenv_path() == repo_root / ".env"


def test_bootstrap_settings_loads_dotenv_once_and_mutates_singleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOCMIND_LOG_LEVEL", raising=False)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (repo_root / ".env").write_text("DOCMIND_LOG_LEVEL=DOTENV\n", encoding="utf-8")

    workdir = repo_root / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    import importlib

    settings_mod = importlib.import_module("src.config.settings")
    current = settings_mod.settings
    before = current.log_level

    try:
        settings_mod.bootstrap_settings(force=True)
        assert current.log_level == "DOTENV"
    finally:
        fresh = settings_mod.DocMindSettings(_env_file=None)  # type: ignore[arg-type]
        settings_mod.apply_settings_in_place(current, fresh)
        settings_mod.reset_bootstrap_state()
        assert current.log_level == before
