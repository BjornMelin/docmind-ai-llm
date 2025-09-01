"""Integration tests for AdaptiveRouterQueryEngine.

Split from tests/test_retrieval/test_query_engine.py to align with the
integration tier. These tests exercise strategy selection, factories, and
realistic query flows using shared fixtures and mocks.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval.query_engine import (
    AdaptiveRouterQueryEngine,
    configure_router_settings,
    create_adaptive_router_engine,
)


@pytest.mark.integration
class TestAdaptiveRouterQueryEngineIntegration:
    """End-to-end behaviors of the router against mocked components."""

    def test_strategy_selection_semantic_query(
        self, mock_vector_index, mock_hybrid_retriever, mock_llm_for_routing
    ):
        """Selects semantic_search for conceptual questions."""
        # Set LLM response text for selector routing
        mock_llm_for_routing.response_text = "semantic_search"
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            llm=mock_llm_for_routing,
        )
        semantic_tool = next(
            t
            for t in router_engine._query_engine_tools
            if t.metadata.name == "semantic_search"
        )
        mock_response = MagicMock(response="Semantic search response")
        semantic_tool.query_engine.query = MagicMock(return_value=mock_response)
        response = router_engine.query(
            "What is the meaning of artificial intelligence?"
        )
        assert response.response == "Semantic search response"

    def test_strategy_selection_hybrid_query(
        self, mock_vector_index, mock_hybrid_retriever, mock_llm_for_routing
    ):
        """Selects hybrid_search for broad/comprehensive queries."""
        mock_llm_for_routing.response_text = "hybrid_search"
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            llm=mock_llm_for_routing,
        )
        # Route through the underlying router, stub its response
        router_engine.router_engine.query = MagicMock(
            return_value=MagicMock(response="Hybrid search response")
        )
        response = router_engine.query(
            "Find documents about machine learning algorithms implementation"
        )
        assert response.response == "Hybrid search response"

    def test_strategy_selection_multi_query_complex(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Selects multi_query_search for complex analytical prompts."""
        mock_llm_for_routing.response_text = "multi_query_search"
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )
        multi_tool = next(
            t
            for t in router_engine._query_engine_tools
            if t.metadata.name == "multi_query_search"
        )
        multi_tool.query_engine.query = MagicMock(
            return_value=MagicMock(response="Multi-query comprehensive response")
        )
        response = router_engine.query(
            "Compare the advantages and disadvantages of different neural "
            "network architectures"
        )
        assert response.response == "Multi-query comprehensive response"


@pytest.mark.integration
class TestAdaptiveRouterQueryEngineFactories:
    """Factory and configuration helper tests."""

    def test_create_adaptive_router_engine_minimal(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Creates a minimal engine with only a vector index."""
        router_engine = create_adaptive_router_engine(
            mock_vector_index, llm=mock_llm_for_routing
        )
        assert isinstance(router_engine, AdaptiveRouterQueryEngine)
        assert router_engine.vector_index == mock_vector_index

    def test_create_adaptive_router_engine_full(
        self,
        mock_vector_index,
        mock_property_graph,
        mock_hybrid_retriever,
        mock_cross_encoder,
        mock_llm_for_routing,
    ):
        """Creates an engine with all optional components configured."""
        mock_multimodal_index = MagicMock()
        mock_multimodal_index.as_query_engine.return_value = MagicMock()
        router_engine = create_adaptive_router_engine(
            vector_index=mock_vector_index,
            kg_index=mock_property_graph,
            hybrid_retriever=mock_hybrid_retriever,
            multimodal_index=mock_multimodal_index,
            reranker=mock_cross_encoder,
            llm=mock_llm_for_routing,
        )
        assert isinstance(router_engine, AdaptiveRouterQueryEngine)

    def test_configure_router_settings(self, mock_vector_index, mock_llm_for_routing):
        """Configures router settings without raising exceptions."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )
        configure_router_settings(router_engine)

    def test_configure_router_settings_failure_handling(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Handles a None router gracefully when configuring settings."""
        AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )
        with patch("src.retrieval.query_engine.logger.error") as mock_error:
            configure_router_settings(None)
            assert not mock_error.called


@pytest.mark.integration
class TestAdaptiveRouterQueryEngineRealism:
    """Realistic scenario and concurrency checks."""

    def test_realistic_query_scenarios(
        self,
        mock_vector_index,
        mock_hybrid_retriever,
        mock_llm_for_routing,
        sample_query_scenarios,
    ):
        """Routes a suite of representative queries with mocked responses."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            llm=mock_llm_for_routing,
        )
        mock_response = MagicMock()
        mock_response.response = "Realistic response"
        router_engine.router_engine.query = MagicMock(return_value=mock_response)
        for scenario in sample_query_scenarios:
            assert (
                router_engine.query(scenario["query"]).response == "Realistic response"
            )

    @pytest.mark.asyncio
    async def test_concurrent_query_execution(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Handles multiple concurrent queries via aquery without contention."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )
        mock_response = MagicMock(response="Concurrent response")
        router_engine.router_engine.aquery = AsyncMock(return_value=mock_response)
        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does attention work?",
            "Compare BERT and GPT",
        ]
        responses = await asyncio.gather(*[router_engine.aquery(q) for q in queries])
        assert len(responses) == len(queries)

    def test_query_truncation_logging(self, mock_vector_index, mock_llm_for_routing):
        """Logs a truncated version of very long queries for readability."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )
        mock_response = MagicMock()
        router_engine.router_engine.query = MagicMock(return_value=mock_response)
        long_query = "This is a very long query " * 10
        with patch("src.retrieval.query_engine.logger.info") as mock_logger:
            router_engine.query(long_query)
        mock_logger.assert_called()
        # Find the call for the truncated query log and validate length
        found = False
        for args, _kwargs in mock_logger.call_args_list:
            if isinstance(args[0], str) and "Executing adaptive query" in args[0]:
                assert len(args[1]) == 100
                found = True
                break
        assert found, "Expected truncated query log was not emitted"
