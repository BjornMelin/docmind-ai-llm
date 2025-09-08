"""Miscellaneous tests for security helpers."""

import pytest

pytestmark = pytest.mark.unit


def test_decrypt_file_passthrough_when_not_enc():
    """Return original path when file suffix is not `.enc`."""
    from src.utils.security import decrypt_file

    assert decrypt_file("/tmp/file.jpg") == "/tmp/file.jpg"


def test_redact_pii_noop():
    """Redact PII stub returns text unchanged."""
    from src.utils.security import redact_pii

    s = "no pii here"
    assert redact_pii(s) == s
