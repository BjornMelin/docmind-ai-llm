"""Unit tests for security utilities.

Covers safe export path validation and log-safety helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import log_safety, security


def test_log_safety_redaction_is_deterministic() -> None:
    """log_safety redaction should be stable and include a fingerprint."""
    redacted_a, fp_a = log_safety.redact_pii(
        "hello@example.com",
        return_fingerprint=True,
    )
    redacted_b, fp_b = log_safety.redact_pii(
        "hello@example.com",
        return_fingerprint=True,
    )
    assert redacted_a == redacted_b
    assert fp_a == fp_b
    assert "hello@example.com" not in redacted_a


def test_validate_export_path_local(tmp_path: Path) -> None:
    """validate_export_path should accept non-egress relative paths under tmp."""
    target = tmp_path / "exports" / "file.txt"
    out = security.validate_export_path(str(target))
    assert out.endswith("file.txt")


def test_validate_export_path_blocks_egress() -> None:
    """validate_export_path must block absolute paths outside the project root."""
    # Force a path that looks like an egress to root
    with pytest.raises(ValueError, match="outside the project root"):
        security.validate_export_path("/etc/passwd")
