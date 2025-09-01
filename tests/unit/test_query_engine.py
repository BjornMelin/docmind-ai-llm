"""Unit tests for AdaptiveRouterQueryEngine components.

Split from tests/test_retrieval/test_query_engine.py to align with the unit
tier. These tests validate construction, tool wiring, and minimal execution
paths using mocks only.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.query_engine import (
    AdaptiveRouterQueryEngine,
)


@pytest.mark.unit
class TestAdaptiveRouterQueryEngineUnit:
    """Basic construction and tool registration for the router engine."""

    @patch("src.retrieval.query_engine.RouterQueryEngine")
    @patch("src.retrieval.query_engine.LLMSingleSelector")
    def test_init_basic_configuration(
        self,
        mock_selector_class,
        mock_router_class,
        mock_vector_index,
        mock_llm_for_routing,
    ):
        """Initializes with minimal components and creates a router engine."""
        mock_selector = MagicMock()
        mock_selector_class.from_defaults.return_value = mock_selector
        mock_router_engine = MagicMock()
        mock_router_class.return_value = mock_router_engine

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        assert router_engine.vector_index == mock_vector_index
        assert router_engine.kg_index is None
        assert router_engine.hybrid_retriever is None
        assert router_engine.multimodal_index is None
        assert router_engine.reranker is None
        mock_router_class.assert_called_once()
        mock_selector_class.from_defaults.assert_called_once_with(
            llm=mock_llm_for_routing
        )

    def test_init_full_configuration(
        self,
        mock_vector_index,
        mock_property_graph,
        mock_hybrid_retriever,
        mock_multimodal_utilities,
        mock_cross_encoder,
        mock_llm_for_routing,
    ):
        """Accepts all optional components and exposes them on the instance."""
        mock_multimodal_index = MagicMock()
        mock_multimodal_index.as_query_engine.return_value = MagicMock()

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            kg_index=mock_property_graph,
            hybrid_retriever=mock_hybrid_retriever,
            multimodal_index=mock_multimodal_index,
            reranker=mock_cross_encoder,
            llm=mock_llm_for_routing,
        )

        assert router_engine.kg_index == mock_property_graph
        assert router_engine.hybrid_retriever == mock_hybrid_retriever
        assert router_engine.multimodal_index == mock_multimodal_index
        assert router_engine.reranker == mock_cross_encoder
        assert len(router_engine._query_engine_tools) >= 3

    def test_create_query_engine_tools_strategy_descriptions(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Describes strategies to guide the selector clearly."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )
        tools = router_engine._query_engine_tools
        semantic_tool = next(t for t in tools if t.metadata.name == "semantic_search")
        assert "semantic" in semantic_tool.metadata.description.lower()
        multi_tool = next(t for t in tools if t.metadata.name == "multi_query_search")
        assert "decomposition" in multi_tool.metadata.description.lower()

    def test_router_creation_failure_with_no_tools(self, mock_llm_for_routing):
        """Current implementation always creates at least one tool; verify fallback."""
        mock_vector_index = MagicMock()
        mock_vector_index.as_query_engine.return_value = None
        engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )
        # Should have at least one strategy tool available
        assert len(engine.get_available_strategies()) >= 1


@pytest.mark.unit
class TestAdaptiveRouterQueryEngineExecution:
    """Execution via query/aquery paths and fallback behavior."""

    def test_query_successful_execution(self, mock_vector_index, mock_llm_for_routing):
        """Returns the router's response when query succeeds."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )
        mock_response = MagicMock()
        mock_response.response = "Test response from router"
        router_engine.router_engine.query = MagicMock(return_value=mock_response)
        response = router_engine.query("What is machine learning?")
        assert response == mock_response

    @pytest.mark.asyncio
    async def test_aquery_successful_execution(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Returns the router's response when aquery succeeds."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )
        mock_response = MagicMock()
        mock_response.response = "Async test response"
        from unittest.mock import AsyncMock

        router_engine.router_engine.aquery = AsyncMock(return_value=mock_response)
        response = await router_engine.aquery("Explain transformers architecture")
        assert response == mock_response

    def test_query_fallback_on_router_failure(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Falls back to vector engine when router raises an error."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )
        router_engine.router_engine.query = MagicMock(side_effect=RuntimeError("fail"))
        mock_fallback = MagicMock()
        router_engine.vector_index.as_query_engine.return_value = mock_fallback
        mock_response = MagicMock()
        mock_fallback.query.return_value = mock_response
        assert router_engine.query("q") == mock_response
