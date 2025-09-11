"""Unit tests for export path validation security functions.

Tests validate_export_path function to ensure it properly blocks:
- Directory traversal attempts (../etc/passwd style attacks)
- Symlink-based attacks
- Path egress outside allowed directories
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils.security import validate_export_path


def test_validate_export_path_blocks_traversal(tmp_path: Path) -> None:
    """Test that directory traversal attempts are blocked.

    Args:
        tmp_path: Temporary directory path provided by pytest fixture.

    This test ensures that attempts to access files outside the allowed
    directory (using ../ patterns) raise a ValueError with appropriate message.
    """
    base = tmp_path / "graph"
    base.mkdir()
    with pytest.raises(ValueError, match="Non-egress export path blocked"):
        validate_export_path(base, "../../etc/passwd")


def test_validate_export_path_blocks_symlink(tmp_path: Path) -> None:
    """Test that symlink-based attacks are blocked.

    Args:
        tmp_path: Temporary directory path provided by pytest fixture.

    This test creates a symlink within the allowed directory and verifies
    that the validation function blocks access to it, preventing potential
    symlink-based security exploits.
    """
    base = tmp_path / "graph"
    base.mkdir()
    target = base / "file.jsonl"
    target.write_text("x")
    link = base / "link.jsonl"
    os.symlink(target, link)
    with pytest.raises(ValueError, match="Symlink export target blocked"):
        validate_export_path(base, "link.jsonl")


def test_validate_export_path_succeeds(tmp_path: Path) -> None:
    """Test that valid export paths are accepted and processed correctly.

    Args:
        tmp_path: Temporary directory path provided by pytest fixture.

    This test verifies that safe, valid paths within the allowed directory
    are accepted, properly resolved, and have their parent directories created
    as expected by the validation function.
    """
    base = tmp_path / "graph"
    base.mkdir()
    p = validate_export_path(base, "dir/graph.jsonl")
    assert str(p).startswith(str(base))
    assert p.parent.exists()
