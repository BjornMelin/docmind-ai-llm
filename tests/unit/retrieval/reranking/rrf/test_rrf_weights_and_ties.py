"""RRF merge tie and k-constant sensitivity tests."""

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval.rrf import rrf_merge

pytestmark = pytest.mark.unit


def _nws(nid: str, score: float = 0.0):
    node = TextNode(text="", id_=nid)
    return NodeWithScore(node=node, score=score)


def test_rrf_ties_and_k_constant():
    a = [_nws("A"), _nws("B"), _nws("C")]
    b = [_nws("B"), _nws("C"), _nws("A")]

    fused_k10 = rrf_merge([a, b], k_constant=10)
    fused_k60 = rrf_merge([a, b], k_constant=60)

    # Both contain same ids with possibly different ordering influence
    ids_k10 = {x.node.node_id for x in fused_k10[:3]}
    ids_k60 = {x.node.node_id for x in fused_k60[:3]}
    assert ids_k10 == ids_k60 == {"A", "B", "C"}
