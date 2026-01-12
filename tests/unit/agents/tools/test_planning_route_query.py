"""Unit tests for planning.route_query (routing heuristics and context).

Split from legacy tests/unit/agents/test_tools.py per migration plan.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from src.agents.tools.planning import route_query

pytestmark = pytest.mark.unit


class TestRouteQuery:
    """Routing heuristics, context handling, and error propagation."""

    def test_route_query_simple_query(self):
        """Simple query routes to vector with high confidence."""
        result_json = route_query.invoke({"query": "Define Python"})
        result = json.loads(result_json)

        assert isinstance(result, dict)
        assert "strategy" in result
        assert "complexity" in result
        assert "needs_planning" in result
        assert "confidence" in result
        assert "processing_time_ms" in result

        assert result["complexity"] in {"simple", "medium"}
        assert result["strategy"] == "vector"
        assert not result["needs_planning"]
        assert result["confidence"] >= 0.9

    def test_route_query_complex_query(self):
        """Complex query routes to hybrid and requires planning."""
        result_json = route_query.invoke({
            "query": (
                "Compare machine learning vs deep learning approaches, analyze "
                "their differences and explain step by step implementation"
            )
        })
        result = json.loads(result_json)

        assert result["complexity"] == "complex"
        assert result["strategy"] == "hybrid"
        assert result["needs_planning"]
        assert result["confidence"] == 0.9
        assert result["word_count"] > 10

    def test_route_query_medium_query(self):
        """Medium query routes to hybrid without planning flag."""
        result_json = route_query.invoke({
            "query": "When was backpropagation introduced"
        })
        result = json.loads(result_json)

        assert result["complexity"] == "medium"
        assert result["strategy"] == "hybrid"
        assert not result["needs_planning"]
        assert result["confidence"] >= 0.8

    def test_route_query_graphrag_patterns(self):
        """GraphRAG indicators flip strategy to graphrag."""
        result_json = route_query.invoke({
            "query": "Show the relationship between concepts and their connections"
        })
        result = json.loads(result_json)

        assert result["strategy"] == "graphrag"

    def test_route_query_with_context(self):
        """Presence of chat history marks context_dependent true."""
        mock_state = {
            "messages": [
                HumanMessage(content="Previous question about AI"),
                HumanMessage(content="Another question"),
                HumanMessage(content="What about this topic?"),
            ]
        }

        result_json = route_query.invoke({
            "query": "What about this topic?",
            "state": mock_state,
        })
        result = json.loads(result_json)

        assert result["context_dependent"]

    def test_route_query_contextual_without_history(self):
        """Contextual phrasing lowers confidence when no history exists."""
        result_json = route_query.invoke({"query": "What about this approach?"})
        result = json.loads(result_json)
        assert result["confidence"] < 0.95

    def test_route_query_error_handling(self):
        """Timer error is propagated to caller for outer recovery tests."""
        # route_query should propagate internal errors
        with (
            patch(
                "src.agents.tools.planning.time.perf_counter",
                side_effect=RuntimeError("Timer error"),
            ),
            pytest.raises(RuntimeError),
        ):
            route_query.invoke({"query": "test query"})

    def test_route_query_boundary_values(self):
        """Word count exact thresholds are reflected in output."""
        result_json = route_query.invoke({
            "query": "This is exactly ten words for medium complexity threshold test"
        })
        result = json.loads(result_json)
        assert result["word_count"] == 10

        long_query = " ".join(["word"] * 20)
        result_json = route_query.invoke({"query": long_query})
        result = json.loads(result_json)
        assert result["word_count"] == 20
