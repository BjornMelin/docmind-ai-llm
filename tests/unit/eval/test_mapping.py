from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.eval.common.mapping import build_doc_mapping, to_doc_id


@pytest.mark.unit
def test_to_doc_id_prefers_metadata_doc_id() -> None:
    """Test that to_doc_id prefers node metadata doc_id."""
    obj = SimpleNamespace(node=SimpleNamespace(metadata={"doc_id": "D123"}))
    assert to_doc_id(obj) == "D123"


@pytest.mark.unit
def test_build_doc_mapping_assigns_ranks() -> None:
    """Test that build_doc_mapping assigns sequential ranks to documents."""

    class _Node:
        def __init__(self, did: str) -> None:
            self.node = SimpleNamespace(metadata={"doc_id": did})

    m = build_doc_mapping({"q1": [_Node("d1"), _Node("d2")]})
    assert m == {"q1": {1: "d1", 2: "d2"}}


@pytest.mark.unit
def test_to_doc_id_fallback_node_doc_id() -> None:
    """Test that to_doc_id falls back to node.doc_id when metadata doc_id absent."""
    obj = SimpleNamespace(node=SimpleNamespace(doc_id="NDOC"))
    assert to_doc_id(obj) == "NDOC"


@pytest.mark.unit
def test_to_doc_id_fallback_node_id_variants() -> None:
    """Test that to_doc_id handles both id_ and id node attributes."""
    obj1 = SimpleNamespace(node=SimpleNamespace(id_="NIDU"))
    obj2 = SimpleNamespace(node=SimpleNamespace(id="NID"))
    assert to_doc_id(obj1) == "NIDU"
    assert to_doc_id(obj2) == "NID"


@pytest.mark.unit
def test_to_doc_id_fallback_top_level_metadata_doc_id() -> None:
    """Test that to_doc_id falls back to top-level metadata doc_id."""
    obj = SimpleNamespace(metadata={"doc_id": "MDOC"})
    assert to_doc_id(obj) == "MDOC"


@pytest.mark.unit
def test_to_doc_id_fallback_top_level_id() -> None:
    """Test that to_doc_id falls back to top-level id attribute."""
    obj = SimpleNamespace(id="OID")
    assert to_doc_id(obj) == "OID"


@pytest.mark.unit
def test_to_doc_id_repr_fallback_when_absent() -> None:
    """Test that to_doc_id falls back to object repr when no id fields present."""

    class Empty:
        def __repr__(self) -> str:  # deterministic repr
            return "<Empty>"

    obj = Empty()
    assert to_doc_id(obj) == "<Empty>"
