"""Regression coverage for the process-global test settings boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import settings

pytestmark = pytest.mark.unit


def test_global_settings_data_root_is_isolated(tmp_path: Path) -> None:
    """Keep process-global writes inside the current test's temp directory."""
    assert settings.data_dir == tmp_path / "data"
