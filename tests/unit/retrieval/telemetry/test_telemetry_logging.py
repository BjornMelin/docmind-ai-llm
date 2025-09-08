"""Telemetry logging tests for AdaptiveRouterQueryEngine.

Verifies that key info/error logs are emitted for routing setup, query
execution, strategy selection, and fallback behavior.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.query_engine import AdaptiveRouterQueryEngine


@pytest.mark.unit
def test_router_engine_logs_creation_and_tool_count(
    mock_vector_index, mock_llm_for_routing
):
    """Logs engine creation and tool count during initialization."""
    with patch("src.retrieval.query_engine.logger") as mock_logger:
        _engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index, llm=mock_llm_for_routing
        )

        # Created tool count and router creation log lines
        info_calls = [str(c.args[0]) for c in mock_logger.info.call_args_list]
        assert any(
            "Created" in msg and "query engine tools" in msg for msg in info_calls
        )
        assert any("RouterQueryEngine created" in msg for msg in info_calls)


@pytest.mark.unit
def test_query_logs_selected_strategy_and_timing(
    mock_vector_index, mock_llm_for_routing
):
    """Logs selected strategy when response metadata includes selector_result."""
    engine = AdaptiveRouterQueryEngine(
        vector_index=mock_vector_index, llm=mock_llm_for_routing
    )
    with patch("src.retrieval.query_engine.logger") as mock_logger:
        # Stub underlying router to return a response with selector_result
        response = MagicMock()
        response.metadata = {"selector_result": "semantic_search"}
        engine.router_engine.query = MagicMock(return_value=response)

        _ = engine.query("what is retrieval?")

        info_calls = [str(c.args[0]) for c in mock_logger.info.call_args_list]
        assert any("Executing adaptive query" in msg for msg in info_calls)
        assert any("Router selected strategy" in msg for msg in info_calls)


@pytest.mark.unit
def test_query_logs_fallback_on_error(mock_vector_index, mock_llm_for_routing):
    """Logs error and fallback message when router raises runtime error."""
    engine = AdaptiveRouterQueryEngine(
        vector_index=mock_vector_index, llm=mock_llm_for_routing
    )
    with patch("src.retrieval.query_engine.logger") as mock_logger:
        engine.router_engine.query = MagicMock(side_effect=RuntimeError("boom"))
        fallback_engine = MagicMock()
        fallback_engine.query = MagicMock(return_value=MagicMock(response="ok"))
        engine.vector_index.as_query_engine.return_value = fallback_engine

        _ = engine.query("q")

        # Verify error and fallback log lines
        error_calls = [str(c.args[0]) for c in mock_logger.error.call_args_list]
        info_calls = [str(c.args[0]) for c in mock_logger.info.call_args_list]
        assert any("RouterQueryEngine failed" in msg for msg in error_calls)
        assert any(
            "Falling back to direct semantic search" in msg for msg in info_calls
        )
