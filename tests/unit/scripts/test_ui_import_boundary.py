"""Regression gate for import-light Streamlit page shells."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_ui_import_boundary_script() -> None:
    repository = Path(__file__).parents[3]
    result = subprocess.run(
        [sys.executable, "scripts/check_ui_import_boundary.py"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "src.app: []",
        "src.pages.01_chat: []",
        "src.pages.02_documents: []",
    ]
