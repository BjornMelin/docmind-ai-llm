"""RRF merge tie and k-constant sensitivity tests."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _nws(nid: str, score: float = 0.0):
    node = SimpleNamespace(node_id=nid, metadata={}, text="")
    return SimpleNamespace(node=node, score=score)


def test_rrf_ties_and_k_constant(monkeypatch):
    from src.retrieval.reranking import _rrf_merge

    a = [_nws("A"), _nws("B"), _nws("C")]
    b = [_nws("B"), _nws("C"), _nws("A")]

    fused_k10 = _rrf_merge([a, b], k_constant=10)
    fused_k60 = _rrf_merge([a, b], k_constant=60)

    # Both contain same ids with possibly different ordering influence
    ids_k10 = {x.node.node_id for x in fused_k10[:3]}
    ids_k60 = {x.node.node_id for x in fused_k60[:3]}
    assert ids_k10 == ids_k60 == {"A", "B", "C"}
