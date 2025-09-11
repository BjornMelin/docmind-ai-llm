"""Unit tests for security utilities.

Covers redact_pii basic behavior and safe export path validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import security


def test_redact_pii_returns_string() -> None:
    """redact_pii should return a string of same type even for empty input."""
    assert isinstance(security.redact_pii("hello"), str)
    assert security.redact_pii("") == ""


def test_validate_export_path_local(tmp_path: Path) -> None:
    """validate_export_path should accept non-egress relative paths under tmp."""
    target = tmp_path / "exports" / "file.txt"
    out = security.validate_export_path(str(target))
    assert out.endswith("file.txt")


def test_validate_export_path_blocks_egress(tmp_path: Path) -> None:
    """validate_export_path must block absolute paths outside the project root."""
    # Force a path that looks like an egress to root
    with pytest.raises(ValueError, match="outside the project root"):
        security.validate_export_path("/etc/passwd")
