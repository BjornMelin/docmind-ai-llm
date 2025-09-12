from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.eval.common.mapping import build_doc_mapping, to_doc_id


@pytest.mark.unit
def test_to_doc_id_prefers_metadata_doc_id() -> None:
    obj = SimpleNamespace(node=SimpleNamespace(metadata={"doc_id": "D123"}))
    assert to_doc_id(obj) == "D123"


@pytest.mark.unit
def test_build_doc_mapping_assigns_ranks() -> None:
    class _Node:
        def __init__(self, did: str) -> None:
            self.node = SimpleNamespace(metadata={"doc_id": did})

    m = build_doc_mapping({"q1": [_Node("d1"), _Node("d2")]})
    assert m == {"q1": {1: "d1", 2: "d2"}}
