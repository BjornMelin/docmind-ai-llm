"""Fallback path tests for AdaptiveRouterQueryEngine.

Covers sub_question_search fallback registration when the SubQuestionQueryEngine
cannot be created (import or construction failure).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.query_engine import AdaptiveRouterQueryEngine


@pytest.mark.unit
def test_sub_question_fallback_registers_tree_summarize(mock_llm_for_routing):
    """Registers sub_question_search fallback when SQE creation fails.

    Ensures the tool is present and that the fallback vector engine is created
    with tree_summarize response mode.
    """
    vec = MagicMock()
    vec.as_query_engine.return_value = MagicMock()

    with patch(
        "src.retrieval.query_engine.SubQuestionQueryEngine.from_defaults",
        side_effect=ImportError("no SQE"),
    ):
        engine = AdaptiveRouterQueryEngine(vector_index=vec, llm=mock_llm_for_routing)

    names = engine.get_available_strategies()
    assert "sub_question_search" in names
    # Verify fallback vector engine creation parameters include tree_summarize
    called_kwargs = vec.as_query_engine.call_args.kwargs
    assert called_kwargs.get("response_mode") == "tree_summarize"
