"""Reciprocal Rank Fusion helpers shared across retrieval modules."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore


def rrf_merge(lists: list[list[NodeWithScore]], k_constant: int) -> list[NodeWithScore]:
    """Rank-level Reciprocal Rank Fusion over multiple reranked lists.

    Args:
        lists: List of ranked lists to merge.
        k_constant: RRF k-constant for score calculation.

    Returns:
        list[NodeWithScore]: Fused list sorted by RRF scores.
    """
    scores: dict[str, tuple[float, NodeWithScore]] = {}
    for ranked in lists:
        for rank, nws in enumerate(ranked, start=1):
            nid = nws.node.node_id
            inc = 1.0 / (k_constant + rank)
            cur = scores.get(nid)
            if cur is None:
                scores[nid] = (inc, nws)
            else:
                scores[nid] = (cur[0] + inc, cur[1])
    fused_list = list(scores.values())
    fused_list.sort(
        key=lambda t: (-float(t[0]), str(getattr(t[1].node, "node_id", "")))
    )
    return [NodeWithScore(node=n.node, score=float(score)) for score, n in fused_list]


__all__ = ["rrf_merge"]
