"""Qdrant helper utilities shared across retrieval and storage modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import islice
from typing import Any

from llama_index.core.schema import NodeWithScore, TextNode

QDRANT_PAYLOAD_FIELDS: tuple[str, ...] = (
    "doc_id",
    "page_id",
    "chunk_id",
    "text",
    "modality",
    "image_path",
)


def normalize_points(result: Any) -> list[Any]:
    """Extract and normalize Qdrant points from query result."""
    points = getattr(result, "points", None) or getattr(result, "result", None) or []
    if points is None:
        return []
    if not isinstance(points, list):
        try:
            points = list(points)
        except TypeError:
            return []
    return points


def order_points(points: Iterable[Any]) -> list[Any]:
    """Return points ordered by score desc then id asc for determinism."""
    return sorted(
        points,
        key=lambda p: (
            -float(getattr(p, "score", 0.0)),
            str(getattr(p, "id", "")),
        ),
    )


def resolve_point_id(
    point: Any,
    payload: dict[str, Any],
    id_keys: Sequence[str],
    *,
    prefer_point_id: bool = True,
) -> str | None:
    """Resolve a node id from point/payload using ordered keys."""
    if prefer_point_id and getattr(point, "id", None) is not None:
        return str(point.id)
    for key in id_keys:
        value = payload.get(key)
        if value:
            return str(value)
    if not prefer_point_id and getattr(point, "id", None) is not None:
        return str(point.id)
    return None


def build_text_nodes(
    points: Iterable[Any],
    *,
    top_k: int,
    id_keys: Sequence[str],
    prefer_point_id: bool = True,
    text_key: str = "text",
) -> list[NodeWithScore]:
    """Build NodeWithScore list from Qdrant points."""
    nodes: list[NodeWithScore] = []
    for idx, point in enumerate(islice(points, int(top_k))):
        payload = getattr(point, "payload", {}) or {}
        score = float(getattr(point, "score", 0.0))
        text = payload.get(text_key) or ""
        node_id = resolve_point_id(
            point, payload, id_keys, prefer_point_id=prefer_point_id
        )
        node = TextNode(text=str(text), id_=(node_id or f"unknown:{idx}"))
        node.metadata.update({k: v for k, v in payload.items() if k != text_key})
        nodes.append(NodeWithScore(node=node, score=score))
    return nodes


def nodes_from_query_result(
    result: Any,
    *,
    top_k: int,
    id_keys: Sequence[str],
    prefer_point_id: bool = True,
    text_key: str = "text",
) -> list[NodeWithScore]:
    """Normalize a query result and build scored text nodes."""
    points = normalize_points(result)
    ordered = order_points(points)
    return build_text_nodes(
        ordered,
        top_k=top_k,
        id_keys=id_keys,
        prefer_point_id=prefer_point_id,
        text_key=text_key,
    )


def get_collection_params(client: Any, collection_name: str) -> Any:
    """Return the collection params object regardless of client shape."""
    info = client.get_collection(collection_name)
    params = getattr(info, "config", info)
    return getattr(params, "params", params)
