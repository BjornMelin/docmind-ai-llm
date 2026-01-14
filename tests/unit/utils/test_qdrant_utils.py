from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.utils.qdrant_utils import (
    build_text_nodes,
    get_collection_params,
    nodes_from_query_result,
    normalize_points,
    order_points,
    resolve_point_id,
)

pytestmark = pytest.mark.unit


@dataclass
class _Point:
    id: object | None = None
    score: float | None = None
    payload: dict[str, object] | None = None


def test_normalize_points_handles_common_shapes() -> None:
    assert normalize_points(SimpleNamespace(points=None)) == []
    assert normalize_points(SimpleNamespace(points=[1, 2])) == [1, 2]
    assert normalize_points(SimpleNamespace(result=(1, 2))) == [1, 2]


def test_normalize_points_rejects_non_iterable() -> None:
    class _Bad:
        points = 123

    assert normalize_points(_Bad()) == []


def test_order_points_is_deterministic() -> None:
    pts = [
        _Point(id="b", score=1.0),
        _Point(id="a", score=1.0),
        _Point(id="c", score=2.0),
    ]
    ordered = order_points(pts)
    assert [p.id for p in ordered] == ["c", "a", "b"]


def test_resolve_point_id_prefers_point_id_then_payload() -> None:
    p = _Point(id="pid", score=1.0, payload={"page_id": "p1"})
    assert resolve_point_id(p, p.payload or {}, ("page_id",)) == "pid"
    assert (
        resolve_point_id(p, p.payload or {}, ("page_id",), prefer_point_id=False)
        == "p1"
    )


def test_build_text_nodes_fills_metadata_and_fallback_ids() -> None:
    points = [
        _Point(
            id=None,
            score=0.5,
            payload={"text": "hello", "doc_id": "d1"},
        ),
        _Point(id="p2", score=0.4, payload={"text": "world"}),
    ]
    nodes = build_text_nodes(points, top_k=10, id_keys=("page_id",), text_key="text")
    assert len(nodes) == 2

    assert nodes[0].score == 0.5
    assert nodes[0].node.text == "hello"
    assert nodes[0].node.metadata["doc_id"] == "d1"
    # first point has no id and payload key not used when
    # prefer_point_id=True -> unknown fallback
    assert nodes[0].node.node_id.startswith("unknown:")

    assert nodes[1].node.node_id == "p2"


def test_nodes_from_query_result_orders_by_score_then_id() -> None:
    result = SimpleNamespace(
        points=[
            _Point(id="b", score=1.0, payload={"text": "b"}),
            _Point(id="a", score=1.0, payload={"text": "a"}),
            _Point(id="c", score=2.0, payload={"text": "c"}),
        ]
    )
    nodes = nodes_from_query_result(result, top_k=10, id_keys=("page_id",))
    assert [n.node.node_id for n in nodes] == ["c", "a", "b"]


def test_get_collection_params_handles_shapes() -> None:
    class _Client:
        def __init__(self, info: object) -> None:
            self._info = info

        def get_collection(self, _name: str) -> object:
            return self._info

    info = SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(x=1)))
    params = get_collection_params(_Client(info), "c")
    assert params.x == 1
