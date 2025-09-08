"""sha256_id stability tests."""

import pytest

pytestmark = pytest.mark.unit


def test_sha256_id_determinism():
    from src.processing.utils import sha256_id

    s = "docmind"
    assert sha256_id(s) == sha256_id(s)
    assert sha256_id(s) != sha256_id(s + "!")
