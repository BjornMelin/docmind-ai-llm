"""RRF merge tie and k-constant sensitivity tests."""

import pytest

from src.retrieval.rrf import rrf_merge

pytestmark = pytest.mark.unit


def test_rrf_ties_and_k_constant(nws_factory):
    a = [nws_factory("A"), nws_factory("B"), nws_factory("C")]
    b = [nws_factory("B"), nws_factory("C"), nws_factory("A")]

    fused_k10 = rrf_merge([a, b], k_constant=10)
    fused_k60 = rrf_merge([a, b], k_constant=60)

    # Both contain same ids with possibly different ordering influence
    ids_k10 = {x.node.node_id for x in fused_k10[:3]}
    ids_k60 = {x.node.node_id for x in fused_k60[:3]}
    assert ids_k10 == ids_k60 == {"A", "B", "C"}
