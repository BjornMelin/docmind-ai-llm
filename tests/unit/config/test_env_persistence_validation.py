"""Unit tests for env persistence validation helpers."""

from __future__ import annotations

import pytest

from src.config.env_persistence import _validate_env_key


@pytest.mark.parametrize(
    "key",
    [
        "",
        "a",
        "1ABC",
        "ABC-DEF",
        "abc",
        "A B",
        "A.B",
        "_ABC",
    ],
)
def test_validate_env_key_rejects_invalid_keys(key: str) -> None:
    with pytest.raises(ValueError, match=r"Invalid env var key"):
        _validate_env_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "A",
        "ABC",
        "A_1",
        "DOCMIND_MODEL",
        "DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS",
    ],
)
def test_validate_env_key_accepts_valid_keys(key: str) -> None:
    _validate_env_key(key)
