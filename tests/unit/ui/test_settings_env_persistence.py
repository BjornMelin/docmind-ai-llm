"""Unit tests for settings .env persistence helper."""

from __future__ import annotations

import pytest
from dotenv import dotenv_values

from src.config.env_persistence import EnvPersistError, persist_env


def test_persist_env_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text("# keep\nEXISTING=1\n", encoding="utf-8")

    persist_env(
        {
            "DOCMIND_MODEL": "Hermes-2-Pro-Llama-3-8B",
            "DOCMIND_LLM_BACKEND": "",
        },
        env_path=env_path,
    )

    values = dotenv_values(env_path)
    assert values.get("EXISTING") == "1"
    assert values.get("DOCMIND_MODEL") == "Hermes-2-Pro-Llama-3-8B"
    assert "DOCMIND_LLM_BACKEND" not in values
    assert "# keep" in env_path.read_text(encoding="utf-8")


def test_persist_env_creates_missing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    env_path = tmp_path / ".env"
    assert not env_path.exists()

    persist_env(
        {"DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS": "false"},
        env_path=env_path,
    )

    values = dotenv_values(env_path)
    assert values.get("DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS") == "false"


def test_persist_env_rejects_control_characters(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    env_path = tmp_path / ".env"

    with pytest.raises(EnvPersistError, match=r"control characters"):
        persist_env({"DOCMIND_MODEL": "line1\nline2"}, env_path=env_path)
