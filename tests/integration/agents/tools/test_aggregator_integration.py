"""Integration smoke tests for src.agents.tools aggregator.

Validates patchability and simple invocation across modules without deep behavior.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.tools.planning import route_query
from src.agents.tools.retrieval import retrieve_documents

pytestmark = pytest.mark.integration


def test_route_query_with_context_recovery():
    """Route with context recovery flags returns decision JSON."""
    mock_state = {
        "context": MagicMock(),
        "context_recovery_enabled": True,
        "reset_context_on_error": True,
    }
    # Should not raise; returns decision JSON
    out = json.loads(
        route_query.invoke({"query": "What about this?", "state": mock_state})
    )
    assert "strategy" in out
    assert "complexity" in out


def test_retrieve_documents_error_json_when_missing_tools():
    """Missing tools_data returns an error JSON structure."""
    out = json.loads(retrieve_documents.invoke({"query": "q", "state": {}}))
    assert out.get("documents") == []
    assert "error" in out


def test_retrieve_documents_with_patched_factory():
    """Patched factory returns mocked documents through vector path."""
    mock_state = {"tools_data": {"vector": MagicMock(), "retriever": None}}
    with patch("src.agents.tool_factory.ToolFactory") as mock_factory:
        mock_tool = MagicMock()
        mock_tool.call.return_value = [{"content": "ok"}]
        mock_factory.create_vector_search_tool.return_value = mock_tool
        out = json.loads(
            retrieve_documents.invoke({
                "query": "q",
                "state": mock_state,
                "strategy": "vector",
            })
        )
    assert out.get("documents")
