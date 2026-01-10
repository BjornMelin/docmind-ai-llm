"""Unit tests for settings .env persistence helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from dotenv import dotenv_values


def _load_settings_module() -> object:
    page_path = (
        Path(__file__).resolve().parents[3] / "src" / "pages" / "04_settings.py"
    )
    spec = importlib.util.spec_from_file_location("settings_page", page_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_persist_env_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text("# keep\nEXISTING=1\n", encoding="utf-8")

    module = _load_settings_module()
    module._persist_env(  # type: ignore[attr-defined]
        {
            "DOCMIND_MODEL": "Hermes-2-Pro-Llama-3-8B",
            "DOCMIND_LLM_BACKEND": "",
        }
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

    module = _load_settings_module()
    module._persist_env(  # type: ignore[attr-defined]
        {"DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS": "false"}
    )

    values = dotenv_values(env_path)
    assert values.get("DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS") == "false"
