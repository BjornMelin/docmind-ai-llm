"""Unit test for RRF merge helper."""

import pytest

from src.retrieval.rrf import rrf_merge

pytestmark = pytest.mark.unit


def test_rrf_merge_prefers_higher_positions(nws_factory):
    """RRF merge should consider rank positions across lists."""
    a = [nws_factory("A", 0.9), nws_factory("B", 0.8), nws_factory("C", 0.7)]
    b = [nws_factory("B", 0.9), nws_factory("C", 0.8), nws_factory("A", 0.7)]
    fused = rrf_merge([a, b], k_constant=60)
    ids = [x.node.node_id for x in fused[:3]]
    assert set(ids) == {"A", "B", "C"}
    assert ids[0] == "B"
