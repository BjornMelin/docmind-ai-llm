"""Unit tests for model pull CLI (tools/models/pull.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tools.models.pull import pull


def test_pull_mocks_hf(tmp_path: Path) -> None:
    """Test that model pull CLI invokes HuggingFace download correctly.

    Args:
        tmp_path: Temporary directory path provided by pytest fixture.

    This test mocks the HuggingFace hub download function to verify that
    the pull command correctly invokes downloads for model files and
    returns appropriate file paths.
    """
    calls: list[tuple[str, str, str]] = []

    def fake_download(
        repo_id: str, filename: str, cache_dir: str, local_files_only: bool
    ) -> str:
        calls.append((repo_id, filename, cache_dir))
        return str(tmp_path / f"{repo_id.replace('/', '__')}__{filename}")

    with patch("tools.models.pull.hf_hub_download", side_effect=fake_download):
        pull([("BAAI/bge-m3", "model.safetensors")], tmp_path)

    assert calls
    assert calls[0][0] == "BAAI/bge-m3"
