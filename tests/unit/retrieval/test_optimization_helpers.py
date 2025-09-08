"""Optimization helpers tests for parse_top_k and related utilities.

Note: Targets pure functions or small helpers only to avoid heavy deps.
"""

import pytest

pytestmark = pytest.mark.unit


def test_parse_top_k_invalid_raises():
    from src.retrieval.optimization import parse_top_k

    with pytest.raises(ValueError, match="must be a positive integer"):
        parse_top_k("-1")
    with pytest.raises(ValueError, match="must be a positive integer"):
        parse_top_k("x")
    # Valid
    assert parse_top_k("10") == 10
