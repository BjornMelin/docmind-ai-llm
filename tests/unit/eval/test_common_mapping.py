"""Unit tests for ``src.eval.common.mapping`` helpers."""

from __future__ import annotations

from types import SimpleNamespace

from src.eval.common.mapping import build_doc_mapping, to_doc_id


class DictNode(dict):
    """Simple dict-like helper to mimic Node metadata."""


def test_to_doc_id_prefers_nested_metadata() -> None:
    node = SimpleNamespace(
        node=SimpleNamespace(metadata={"doc_id": "meta-123"}),
        metadata={"doc_id": "fallback"},
        id="ignored",
    )
    assert to_doc_id(node) == "meta-123"


def test_to_doc_id_falls_back_to_repr() -> None:
    node = SimpleNamespace()
    assert to_doc_id(node) == repr(node)


def test_build_doc_mapping_assigns_ranks() -> None:
    node_a = DictNode(node=SimpleNamespace(doc_id="A"))
    node_b = DictNode(id="B")
    mapping = build_doc_mapping({"q1": [node_a, node_b]})
    assert mapping == {"q1": {1: "A", 2: "B"}}
