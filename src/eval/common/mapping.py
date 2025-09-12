"""Doc ID mapping helpers for evaluation reproducibility.

These utilities convert retrieval results to canonical BEIR doc ids and
persist rank mappings for reproducibility/debugging.
"""

from __future__ import annotations

from typing import Any


def to_doc_id(node_like: Any) -> str:
    """Extract a document identifier from a retrieval node-like object.

    The function checks common locations where a "doc_id" may appear:
        - ``obj.node.metadata['doc_id']``
        - ``obj.node.doc_id``
        - ``obj.node.id_`` / ``obj.node.id`` as fallback
        - ``obj.metadata['doc_id']``
        - ``obj.id`` as last resort
    """
    # NodeWithScore.node.metadata path
    try:
        md = getattr(getattr(node_like, "node", None), "metadata", None) or {}
        if isinstance(md, dict) and "doc_id" in md and md["doc_id"] is not None:
            return str(md["doc_id"])
    except Exception:  # pragma: no cover - defensive
        # fall through to other strategies
        ...

    # NodeWithScore.node direct attributes
    try:
        node = getattr(node_like, "node", None)
        for attr in ("doc_id", "id_", "id"):
            val = getattr(node, attr, None)
            if val is not None:
                return str(val)
    except Exception:  # pragma: no cover - defensive
        ...

    # Flat metadata
    try:
        md = getattr(node_like, "metadata", None) or {}
        if isinstance(md, dict) and md.get("doc_id") is not None:
            return str(md["doc_id"])
    except Exception:  # pragma: no cover - defensive
        ...

    # Final fallback: object id attribute or repr
    val = getattr(node_like, "id", None)
    return str(val) if val is not None else str(node_like)


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
