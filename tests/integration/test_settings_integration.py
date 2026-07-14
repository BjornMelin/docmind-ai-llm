"""Integration tests for the typed settings and filesystem contract."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config.settings import DocMindSettings

pytestmark = pytest.mark.integration


@pytest.fixture(name="settings_env")
def fixture_settings_env(tmp_path: Path) -> dict[str, str]:
    """Return isolated environment values for settings integration tests."""
    return {
        "DOCMIND_DATA_DIR": str(tmp_path / "data"),
        "DOCMIND_CACHE__DIR": str(tmp_path / "cache"),
        "DOCMIND_LOG_FILE": str(tmp_path / "logs" / "test.log"),
        "DOCMIND_DEBUG": "true",
    }


def test_environment_paths_support_real_file_io(settings_env: dict[str, str]) -> None:
    """Environment-derived paths are typed and usable for application data."""
    with patch.dict(os.environ, settings_env):
        config = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    assert config.data_dir == Path(settings_env["DOCMIND_DATA_DIR"])
    assert config.cache.dir == Path(settings_env["DOCMIND_CACHE__DIR"])
    assert config.log_file == Path(settings_env["DOCMIND_LOG_FILE"])
    assert config.debug is True

    config.data_dir.mkdir(parents=True)
    config.cache.dir.mkdir(parents=True)
    data_file = config.data_dir / "integration.txt"
    cache_file = config.cache.dir / "entry.json"
    data_file.write_text("DocMind", encoding="utf-8")
    cache_file.write_text('{"ready": true}', encoding="utf-8")

    assert data_file.read_text(encoding="utf-8") == "DocMind"
    assert cache_file.read_text(encoding="utf-8") == '{"ready": true}'


def test_serialization_preserves_nested_path_types(
    settings_env: dict[str, str],
) -> None:
    """Pydantic serialization preserves the nested settings contract."""
    with patch.dict(os.environ, settings_env):
        config = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    payload = config.model_dump()
    assert payload["data_dir"] == config.data_dir
    assert payload["cache"]["dir"] == config.cache.dir
    assert payload["agents"]["decision_timeout"] == config.agents.decision_timeout


def test_settings_are_consistent_across_threads(settings_env: dict[str, str]) -> None:
    """Independent settings instances read one consistent environment contract."""

    def load() -> tuple[bool, Path, Path]:
        config = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
        return config.debug, config.data_dir, config.cache.dir

    with (
        patch.dict(os.environ, settings_env),
        ThreadPoolExecutor(max_workers=5) as pool,
    ):
        results = list(pool.map(lambda _: load(), range(10)))

    assert len(set(results)) == 1
    assert results[0] == (
        True,
        Path(settings_env["DOCMIND_DATA_DIR"]),
        Path(settings_env["DOCMIND_CACHE__DIR"]),
    )


def test_invalid_environment_values_fail_closed(settings_env: dict[str, str]) -> None:
    """Invalid environment values raise instead of silently falling back."""
    invalid_env = {**settings_env, "DOCMIND_DEBUG": "not-a-boolean"}

    with patch.dict(os.environ, invalid_env), pytest.raises(ValidationError):
        DocMindSettings(_env_file=None)  # type: ignore[arg-type]
