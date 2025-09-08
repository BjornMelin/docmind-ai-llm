"""Tests for ToolFactory combinations and ordering.

Google-Style Docstrings:
    Verifies that ToolFactory creates tools with the expected ordering and
    includes optional keyword tool when enabled.
"""

import pytest

pytestmark = pytest.mark.unit


def test_tool_factory_ordering_and_keyword(monkeypatch):
    """Ensure ordering: hybrid (retriever or vector), KG optional, vector, keyword.

    Mocks a simple index that returns an object from `as_query_engine` and
    sets the keyword flag to True to include the keyword tool at the end.
    """
    from src.agents.tool_factory import ToolFactory
    from src.config import settings as app_settings

    class _Idx:
        def as_query_engine(self, **_):
            return object()

    class _Retriever:
        pass

    # Enable keyword tool
    monkeypatch.setattr(
        app_settings.retrieval, "enable_keyword_tool", True, raising=False
    )
    tools = ToolFactory.create_tools_from_indexes(
        vector_index=_Idx(), kg_index=_Idx(), retriever=_Retriever()
    )
    names = [t.metadata.name for t in tools]
    # Starts with hybrid fusion search when retriever present, includes kg and
    # vector, ends with keyword
    assert names[0] in {"hybrid_search", "hybrid_vector_search"}
    assert "vector_search" in names
    assert any("knowledge" in n for n in names)
    assert names[-1] == "keyword_search"
