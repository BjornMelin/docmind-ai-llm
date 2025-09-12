"""Doc ID mapping helpers for evaluation reproducibility.

These utilities convert retrieval results to canonical BEIR doc ids and
persist rank mappings for reproducibility/debugging.
"""

from __future__ import annotations

from typing import Any


def to_doc_id(node_like: Any) -> str:
    """Extract a document identifier from a retrieval node-like object.

    Tries a sequence of attribute paths in order, then falls back to repr().
    """
    lookup_paths = [
        ("node", "metadata", "doc_id"),
        ("node", "doc_id"),
        ("node", "id_"),
        ("node", "id"),
        ("metadata", "doc_id"),
        ("id",),
    ]
    for path in lookup_paths:
        val = node_like
        for attr in path:
            val = val.get(attr) if isinstance(val, dict) else getattr(val, attr, None)
            if val is None:
                break
        else:
            return str(val)
    return repr(node_like)


def build_doc_mapping(results: dict[str, list[Any]]) -> dict[str, dict[int, str]]:
    """Build per-query rank→doc_id mapping for reproducibility.

    Args:
        results: Mapping from query-id to a list of NodeWithScore-like objects,
            each convertible to ``doc_id`` via :func:`to_doc_id`.

    Returns:
        A mapping: ``{qid: {rank: doc_id}}`` where ranks start at 1.
    """
    mapping: dict[str, dict[int, str]] = {}
    for qid, nodes in results.items():
        rank_map: dict[int, str] = {}
        for idx, n in enumerate(nodes, start=1):
            rank_map[idx] = to_doc_id(n)
        mapping[qid] = rank_map
    return mapping


__all__ = ["build_doc_mapping", "to_doc_id"]
