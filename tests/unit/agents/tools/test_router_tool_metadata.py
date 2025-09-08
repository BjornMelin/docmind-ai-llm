"""Router tool metadata shape tests."""

import pytest

pytestmark = pytest.mark.unit


def test_router_tool_metadata_shapes():
    from src.agents.tool_factory import ToolFactory

    class _Idx:
        def as_query_engine(self, **_):
            return object()

    tools = ToolFactory.create_basic_tools({"vector": _Idx()})
    assert isinstance(tools, list)
    assert len(tools) >= 1
    md = getattr(tools[0], "metadata", None)
    assert md is not None
    assert hasattr(md, "name")
    assert hasattr(md, "description")
