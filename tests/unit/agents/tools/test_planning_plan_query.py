"""Unit tests for planning.plan_query (query decomposition).

Split from legacy tests/unit/agents/test_tools.py per migration plan.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.agents.tools.planning import plan_query

pytestmark = pytest.mark.unit


class TestPlanQuery:
    """Plan decomposition scenarios and fallback behavior."""

    def test_plan_query_simple(self):
        """Simple complexity passes query through without decomposition."""
        result_json = plan_query.invoke(
            {
                "query": "What is AI?",
                "complexity": "simple",
            }
        )
        result = json.loads(result_json)

        assert isinstance(result, dict)
        assert "original_query" in result
        assert "sub_tasks" in result
        assert "execution_order" in result
        assert "estimated_complexity" in result
        assert result["sub_tasks"] == ["What is AI?"]
        assert result["execution_order"] == "sequential"

    def test_plan_query_comparison(self):
        """Comparison queries produce parallel tasks and entity coverage."""
        result_json = plan_query.invoke(
            {
                "query": "Compare AI vs ML performance",
                "complexity": "complex",
            }
        )
        result = json.loads(result_json)

        assert len(result["sub_tasks"]) >= 3
        assert result["execution_order"] == "parallel"
        assert any("AI" in task for task in result["sub_tasks"])
        assert any("ML" in task for task in result["sub_tasks"])
        assert any("Compare" in task for task in result["sub_tasks"])

    def test_plan_query_analysis(self):
        """Analysis queries include components/background subtasks."""
        result_json = plan_query.invoke(
            {
                "query": "Analyze the performance of neural networks",
                "complexity": "complex",
            }
        )
        result = json.loads(result_json)

        assert len(result["sub_tasks"]) >= 3
        text = " ".join(result["sub_tasks"]).lower()
        assert ("components" in text) or ("background" in text)

    def test_plan_query_process_explanation(self):
        """Process/how-to queries include steps/definition subtasks."""
        result_json = plan_query.invoke(
            {
                "query": "How does gradient descent work?",
                "complexity": "complex",
            }
        )
        result = json.loads(result_json)

        assert len(result["sub_tasks"]) >= 3
        text = " ".join(result["sub_tasks"]).lower()
        assert ("steps" in text) or ("definition" in text)

    def test_plan_query_list_enumeration(self):
        """Enumeration queries include list/categorize subtasks."""
        result_json = plan_query.invoke(
            {
                "query": "List the types of machine learning algorithms",
                "complexity": "complex",
            }
        )
        result = json.loads(result_json)

        assert len(result["sub_tasks"]) >= 2
        text = " ".join(result["sub_tasks"]).lower()
        assert ("list" in text) or ("categorize" in text)

    def test_plan_query_default_decomposition(self):
        """Default path provides generic multi-step decomposition."""
        result_json = plan_query.invoke(
            {
                "query": "Complex query with multiple aspects and various components",
                "complexity": "complex",
            }
        )
        result = json.loads(result_json)

        assert len(result["sub_tasks"]) >= 1
        assert result["task_count"] == len(result["sub_tasks"])
        assert result["estimated_complexity"] in ["medium", "high"]

    def test_plan_query_error_handling(self):
        """Timer error triggers fallback, not an exception."""
        with patch(
            "src.agents.tools.planning.time.perf_counter",
            side_effect=RuntimeError("Timer error"),
        ):
            result_json = plan_query.invoke(
                {
                    "query": "test query",
                    "complexity": "complex",
                }
            )
            result = json.loads(result_json)
            assert "error" in result
            assert result["sub_tasks"] == ["test query"]
            assert result["execution_order"] == "sequential"
