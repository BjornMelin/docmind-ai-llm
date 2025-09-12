"""Basic tests for ToolFactory creation paths.

Validates that ToolFactory creates QueryEngineTool objects with expected
names and does not require heavy dependencies.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agents.tool_factory import ToolFactory


class _IndexStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def as_query_engine(self, **kwargs: Any) -> str:  # pragma: no cover - trivial
        self.calls.append(("as_query_engine", kwargs))
        return "qe"


@pytest.mark.unit
def test_create_vector_search_tool_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid heavy reranking by returning empty postprocessors
    import src.retrieval.reranking as rr

    monkeypatch.setattr(rr, "get_postprocessors", lambda *_a, **_k: [], raising=True)
    idx = _IndexStub()
    tool = ToolFactory.create_vector_search_tool(idx)
    assert tool.metadata.name == "vector_search"
    # Ensure index.as_query_engine was called
    assert idx.calls
    assert idx.calls[-1][0] == "as_query_engine"


@pytest.mark.unit
def test_create_keyword_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.retrieval.reranking as rr

    monkeypatch.setattr(rr, "get_postprocessors", lambda *_a, **_k: [], raising=True)
    idx = _IndexStub()
    tool = ToolFactory.create_keyword_tool(idx)
    assert tool.metadata.name == "keyword_search"


class _KGIndexStub:
    def as_query_engine(self, **kwargs: Any) -> str:  # pragma: no cover - trivial
        return "kgqe"


@pytest.mark.unit
def test_create_kg_search_tool() -> None:
    kg = _KGIndexStub()
    tool = ToolFactory.create_kg_search_tool(kg)
    assert tool is not None
    assert tool.metadata.name == "knowledge_graph"
