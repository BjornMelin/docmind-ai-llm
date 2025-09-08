"""Unit test for RRF merge helper in reranking module."""

from types import SimpleNamespace

from src.retrieval.reranking import _rrf_merge


def _nws(nid: str, score: float):
    node = SimpleNamespace(node_id=nid, metadata={}, text="")
    return SimpleNamespace(node=node, score=score)


def test_rrf_merge_prefers_higher_positions():
    """RRF merge should consider rank positions across lists."""
    a = [_nws("A", 0.9), _nws("B", 0.8), _nws("C", 0.7)]
    b = [_nws("B", 0.9), _nws("C", 0.8), _nws("A", 0.7)]
    fused = _rrf_merge([a, b], k_constant=60)
    ids = [x.node.node_id for x in fused[:3]]
    assert set(ids) == {"A", "B", "C"}
