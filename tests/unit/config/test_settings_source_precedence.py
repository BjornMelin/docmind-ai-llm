"""Tests for Pydantic Settings source precedence (env vs dotenv vs init)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import DocMindSettings

pytestmark = pytest.mark.unit


def test_settings_source_precedence_init_env_dotenv_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Precedence: init kwargs > env vars > dotenv (when opted in) > defaults."""
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("DOCMIND_LOG_LEVEL=DOTENV\n", encoding="utf-8")

    # Without explicit `_env_file`, dotenv is not loaded.
    assert DocMindSettings().log_level == "INFO"

    # Dotenv opt-in via `_env_file`.
    assert DocMindSettings(_env_file=env_file).log_level == "DOTENV"  # type: ignore[arg-type]

    # env overrides dotenv (when opted in)
    monkeypatch.setenv("DOCMIND_LOG_LEVEL", "ENV")
    assert DocMindSettings(_env_file=env_file).log_level == "ENV"  # type: ignore[arg-type]

    # init overrides env
    assert DocMindSettings(log_level="INIT").log_level == "INIT"

    # default when dotenv is disabled (explicit)
    monkeypatch.delenv("DOCMIND_LOG_LEVEL", raising=False)
    assert DocMindSettings(_env_file=None).log_level == "INFO"  # type: ignore[arg-type]
