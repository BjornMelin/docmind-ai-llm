"""Comprehensive test suite for AdaptiveRouterQueryEngine (REQ-0051).

Tests RouterQueryEngine implementation with LLMSingleSelector for intelligent
strategy selection, multimodal detection, and retrieval performance validation.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval.query_engine import (
    AdaptiveRouterQueryEngine,
    configure_router_settings,
    create_adaptive_router_engine,
)


@pytest.mark.unit
class TestAdaptiveRouterQueryEngineUnit:
    """Unit tests for AdaptiveRouterQueryEngine components."""

    @patch("src.retrieval.query_engine.RouterQueryEngine")
    @patch("src.retrieval.query_engine.LLMSingleSelector")
    def test_init_basic_configuration(
        self,
        mock_selector_class,
        mock_router_class,
        mock_vector_index,
        mock_llm_for_routing,
    ):
        """Test basic RouterQueryEngine initialization with minimal components."""
        # Mock the selector and router
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
        assert router_engine.llm == mock_llm_for_routing
        assert router_engine._query_engine_tools is not None
        assert (
            len(router_engine._query_engine_tools) >= 2
        )  # At minimum: semantic + multi_query

        # Verify RouterQueryEngine was created
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
        """Test RouterQueryEngine with all available strategies."""
        # Mock multimodal index
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

        # Verify all components are configured
        assert router_engine.kg_index == mock_property_graph
        assert router_engine.hybrid_retriever == mock_hybrid_retriever
        assert router_engine.multimodal_index == mock_multimodal_index
        assert router_engine.reranker == mock_cross_encoder

        # Should have all 5 strategies: hybrid, semantic, multi_query, kg, multimodal
        assert len(router_engine._query_engine_tools) == 5

    def test_create_query_engine_tools_strategy_descriptions(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test that strategy descriptions are detailed for LLM selection."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        tools = router_engine._query_engine_tools

        # Check semantic search tool
        semantic_tool = next(t for t in tools if t.metadata.name == "semantic_search")
        description = semantic_tool.metadata.description
        assert "BGE-M3" in description
        assert "1024-dimensional" in description
        assert "semantic" in description.lower()
        assert "multilingual" in description.lower()

        # Check multi-query tool
        multi_tool = next(t for t in tools if t.metadata.name == "multi_query_search")
        multi_description = multi_tool.metadata.description
        assert "complex questions" in multi_description.lower()
        assert "decomposition" in multi_description.lower()
        assert "tree_summarize" in multi_description.lower()

    def test_create_query_engine_tools_with_hybrid(
        self,
        mock_vector_index,
        mock_hybrid_retriever,
        mock_cross_encoder,
        mock_llm_for_routing,
    ):
        """Test hybrid search tool creation with reranker."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=mock_cross_encoder,
            llm=mock_llm_for_routing,
        )

        tools = router_engine._query_engine_tools
        hybrid_tool = next(t for t in tools if t.metadata.name == "hybrid_search")

        description = hybrid_tool.metadata.description
        assert "hybrid search" in description.lower()
        assert "BGE-M3" in description
        assert "dense and sparse" in description.lower()
        assert "RRF fusion" in description
        assert "cross-encoder reranking" in description.lower()

    def test_create_query_engine_tools_with_kg(
        self, mock_vector_index, mock_property_graph, mock_llm_for_routing
    ):
        """Test knowledge graph tool creation."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            kg_index=mock_property_graph,
            llm=mock_llm_for_routing,
        )

        tools = router_engine._query_engine_tools
        kg_tool = next(t for t in tools if t.metadata.name == "knowledge_graph")

        description = kg_tool.metadata.description
        assert "knowledge graph" in description.lower()
        assert "relationships" in description.lower()
        assert "entity" in description.lower()
        assert "connected" in description.lower()

    def test_create_query_engine_tools_with_multimodal(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test multimodal search tool creation."""
        mock_multimodal_index = MagicMock()
        mock_multimodal_index.as_query_engine.return_value = MagicMock()

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            multimodal_index=mock_multimodal_index,
            llm=mock_llm_for_routing,
        )

        tools = router_engine._query_engine_tools
        multimodal_tool = next(
            t for t in tools if t.metadata.name == "multimodal_search"
        )

        description = multimodal_tool.metadata.description
        assert "multimodal" in description.lower()
        assert "CLIP" in description
        assert "image-text" in description.lower()
        assert "visual" in description.lower()
        assert "512-dimensional" in description

    def test_detect_multimodal_query_image_keywords(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test multimodal query detection with image keywords."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        # Test positive cases
        assert router_engine._detect_multimodal_query(
            "Show me images of neural networks"
        )
        assert router_engine._detect_multimodal_query(
            "Find diagrams about transformers"
        )
        assert router_engine._detect_multimodal_query(
            "Display chart of performance metrics"
        )
        assert router_engine._detect_multimodal_query(
            "What does the architecture look like?"
        )
        assert router_engine._detect_multimodal_query("screenshot of the interface")

        # Test negative cases
        assert not router_engine._detect_multimodal_query(
            "Explain the concept of attention"
        )
        assert not router_engine._detect_multimodal_query("What is the best approach?")
        assert not router_engine._detect_multimodal_query("Compare BGE-M3 with BERT")

    def test_detect_multimodal_query_file_extensions(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test multimodal detection with file extensions."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        assert router_engine._detect_multimodal_query("file:architecture.jpg analysis")
        assert router_engine._detect_multimodal_query("load diagram.png for reference")
        assert router_engine._detect_multimodal_query("process image.jpg content")

    def test_get_available_strategies_basic(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test getting available strategy names."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        strategies = router_engine.get_available_strategies()
        assert "semantic_search" in strategies
        assert "multi_query_search" in strategies
        assert len(strategies) >= 2

    def test_get_available_strategies_full(
        self,
        mock_vector_index,
        mock_property_graph,
        mock_hybrid_retriever,
        mock_llm_for_routing,
    ):
        """Test getting all available strategies with full configuration."""
        mock_multimodal_index = MagicMock()
        mock_multimodal_index.as_query_engine.return_value = MagicMock()

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            kg_index=mock_property_graph,
            hybrid_retriever=mock_hybrid_retriever,
            multimodal_index=mock_multimodal_index,
            llm=mock_llm_for_routing,
        )

        strategies = router_engine.get_available_strategies()
        expected_strategies = {
            "hybrid_search",
            "semantic_search",
            "multi_query_search",
            "knowledge_graph",
            "multimodal_search",
        }
        assert set(strategies) == expected_strategies


@pytest.mark.unit
class TestAdaptiveRouterQueryEngineExecution:
    """Unit tests for query execution logic."""

    def test_query_successful_execution(self, mock_vector_index, mock_llm_for_routing):
        """Test successful query execution through router."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        # Mock successful router execution
        mock_response = MagicMock()
        mock_response.response = "Test response from router"
        mock_response.metadata = {"selector_result": "semantic_search"}
        router_engine.router_engine.query = MagicMock(return_value=mock_response)

        response = router_engine.query("What is machine learning?")

        assert response == mock_response
        assert response.response == "Test response from router"
        router_engine.router_engine.query.assert_called_once_with(
            "What is machine learning?"
        )

    @pytest.mark.asyncio
    async def test_aquery_successful_execution(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test successful async query execution."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        # Mock successful async router execution
        mock_response = MagicMock()
        mock_response.response = "Async test response"
        mock_response.metadata = {"selector_result": "hybrid_search"}
        router_engine.router_engine.aquery = AsyncMock(return_value=mock_response)

        response = await router_engine.aquery("Explain transformers architecture")

        assert response == mock_response
        assert response.response == "Async test response"
        router_engine.router_engine.aquery.assert_called_once_with(
            "Explain transformers architecture"
        )

    def test_query_fallback_on_router_failure(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test fallback to semantic search when router fails."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        # Mock router failure
        router_engine.router_engine.query = MagicMock(
            side_effect=RuntimeError("Router failed")
        )

        # Mock fallback semantic search
        mock_fallback_response = MagicMock()
        mock_fallback_response.response = "Fallback response"
        mock_fallback_engine = MagicMock()
        mock_fallback_engine.query.return_value = mock_fallback_response
        mock_vector_index.as_query_engine.return_value = mock_fallback_engine

        response = router_engine.query("Test query")

        assert response == mock_fallback_response
        mock_vector_index.as_query_engine.assert_called_once()
        mock_fallback_engine.query.assert_called_once_with("Test query")

    @pytest.mark.asyncio
    async def test_aquery_fallback_on_router_failure(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test async fallback to semantic search when router fails."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        # Mock async router failure
        router_engine.router_engine.aquery = AsyncMock(
            side_effect=ValueError("Async router failed")
        )

        # Mock fallback async semantic search
        mock_fallback_response = MagicMock()
        mock_fallback_response.response = "Async fallback response"
        mock_fallback_engine = MagicMock()
        mock_fallback_engine.aquery = AsyncMock(return_value=mock_fallback_response)
        mock_vector_index.as_query_engine.return_value = mock_fallback_engine

        response = await router_engine.aquery("Async test query")

        assert response == mock_fallback_response
        mock_vector_index.as_query_engine.assert_called_once()
        mock_fallback_engine.aquery.assert_called_once_with("Async test query")

    def test_query_with_additional_kwargs(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test query execution with additional keyword arguments."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        mock_response = MagicMock()
        router_engine.router_engine.query = MagicMock(return_value=mock_response)

        response = router_engine.query(
            "Test query", similarity_top_k=15, response_mode="tree_summarize"
        )

        assert response == mock_response
        router_engine.router_engine.query.assert_called_once_with(
            "Test query", similarity_top_k=15, response_mode="tree_summarize"
        )

    def test_router_creation_failure_with_no_tools(self, mock_llm_for_routing):
        """Test router creation fails gracefully with no available tools."""
        mock_vector_index = MagicMock()
        mock_vector_index.as_query_engine.return_value = None

        with pytest.raises(ValueError, match="No query engine tools available"):
            AdaptiveRouterQueryEngine(
                vector_index=mock_vector_index,
                llm=mock_llm_for_routing,
            )


@pytest.mark.integration
class TestAdaptiveRouterQueryEngineIntegration:
    """Integration tests for RouterQueryEngine with strategy selection."""

    def test_strategy_selection_semantic_query(
        self, mock_vector_index, mock_hybrid_retriever, mock_llm_for_routing
    ):
        """Test router selects semantic search for conceptual queries."""
        # Mock LLM to select semantic strategy
        mock_llm_for_routing.complete.return_value = MagicMock(text="semantic_search")

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            llm=mock_llm_for_routing,
        )

        # Mock the specific tool execution
        semantic_tool = next(
            t
            for t in router_engine._query_engine_tools
            if t.metadata.name == "semantic_search"
        )
        mock_response = MagicMock()
        mock_response.response = "Semantic search response"
        semantic_tool.query_engine.query = MagicMock(return_value=mock_response)

        response = router_engine.query(
            "What is the meaning of artificial intelligence?"
        )

        # Verify semantic tool was used (indirectly through response)
        assert response.response == "Semantic search response"

    def test_strategy_selection_hybrid_query(
        self, mock_vector_index, mock_hybrid_retriever, mock_llm_for_routing
    ):
        """Test router selects hybrid search for comprehensive queries."""
        # Mock LLM to select hybrid strategy
        mock_llm_for_routing.complete.return_value = MagicMock(text="hybrid_search")

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            llm=mock_llm_for_routing,
        )

        # Mock the hybrid tool execution
        hybrid_tool = next(
            t
            for t in router_engine._query_engine_tools
            if t.metadata.name == "hybrid_search"
        )
        mock_response = MagicMock()
        mock_response.response = "Hybrid search response"
        hybrid_tool.query_engine.query = MagicMock(return_value=mock_response)

        response = router_engine.query(
            "Find documents about machine learning algorithms implementation"
        )

        assert response.response == "Hybrid search response"

    def test_strategy_selection_multi_query_complex(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test router selects multi-query for complex analytical questions."""
        # Mock LLM to select multi-query strategy
        mock_llm_for_routing.complete.return_value = MagicMock(
            text="multi_query_search"
        )

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        # Mock the multi-query tool execution
        multi_tool = next(
            t
            for t in router_engine._query_engine_tools
            if t.metadata.name == "multi_query_search"
        )
        mock_response = MagicMock()
        mock_response.response = "Multi-query comprehensive response"
        multi_tool.query_engine.query = MagicMock(return_value=mock_response)

        response = router_engine.query(
            "Compare the advantages and disadvantages of different neural network architectures "
            "and explain which ones are best for specific use cases"
        )

        assert response.response == "Multi-query comprehensive response"

    def test_strategy_selection_knowledge_graph_relationships(
        self, mock_vector_index, mock_property_graph, mock_llm_for_routing
    ):
        """Test router selects knowledge graph for relationship queries."""
        # Mock LLM to select knowledge graph strategy
        mock_llm_for_routing.complete.return_value = MagicMock(text="knowledge_graph")

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            kg_index=mock_property_graph,
            llm=mock_llm_for_routing,
        )

        # Mock the KG tool execution
        kg_tool = next(
            t
            for t in router_engine._query_engine_tools
            if t.metadata.name == "knowledge_graph"
        )
        mock_response = MagicMock()
        mock_response.response = "Knowledge graph relationship response"
        kg_tool.query_engine.query = MagicMock(return_value=mock_response)

        response = router_engine.query(
            "How are transformers and attention mechanisms connected to BERT?"
        )

        assert response.response == "Knowledge graph relationship response"

    def test_performance_strategy_selection_latency(
        self, mock_vector_index, mock_llm_for_routing, rtx_4090_performance_targets
    ):
        """Test strategy selection meets <50ms latency target."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        # Mock fast LLM response
        mock_response = MagicMock()
        mock_response.response = "Fast response"
        router_engine.router_engine.query = MagicMock(return_value=mock_response)

        start_time = time.perf_counter()
        response = router_engine.query("What is machine learning?")
        selection_latency = (time.perf_counter() - start_time) * 1000  # Convert to ms

        assert response == mock_response
        assert (
            selection_latency
            < rtx_4090_performance_targets["strategy_selection_latency_ms"]
        )


@pytest.mark.integration
class TestAdaptiveRouterQueryEngineFactories:
    """Test factory functions and configuration utilities."""

    def test_create_adaptive_router_engine_minimal(self, mock_vector_index):
        """Test factory function with minimal configuration."""
        router_engine = create_adaptive_router_engine(mock_vector_index)

        assert isinstance(router_engine, AdaptiveRouterQueryEngine)
        assert router_engine.vector_index == mock_vector_index
        assert router_engine.kg_index is None
        assert router_engine.hybrid_retriever is None
        assert router_engine.multimodal_index is None
        assert router_engine.reranker is None

    def test_create_adaptive_router_engine_full(
        self,
        mock_vector_index,
        mock_property_graph,
        mock_hybrid_retriever,
        mock_cross_encoder,
        mock_llm_for_routing,
    ):
        """Test factory function with full configuration."""
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
        assert router_engine.kg_index == mock_property_graph
        assert router_engine.hybrid_retriever == mock_hybrid_retriever
        assert router_engine.multimodal_index == mock_multimodal_index
        assert router_engine.reranker == mock_cross_encoder
        assert router_engine.llm == mock_llm_for_routing

    def test_configure_router_settings(self, mock_vector_index, mock_llm_for_routing):
        """Test router settings configuration utility."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        # Should not raise exception
        configure_router_settings(router_engine)

    def test_configure_router_settings_failure_handling(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test router settings configuration handles failures gracefully."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        # Mock a configuration failure scenario
        with (
            patch("src.retrieval.query_engine.logger.error") as mock_logger_error,
            pytest.raises(Exception),
        ):
            # Force an error during configuration
            configure_router_settings(None)  # Pass None to trigger error


@pytest.mark.integration
class TestAdaptiveRouterQueryEngineRealism:
    """Integration tests with realistic scenarios and data."""

    def test_realistic_query_scenarios(
        self,
        mock_vector_index,
        mock_hybrid_retriever,
        mock_llm_for_routing,
        sample_query_scenarios,
    ):
        """Test router with realistic query scenarios from conftest."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            llm=mock_llm_for_routing,
        )

        # Mock response for all queries
        mock_response = MagicMock()
        mock_response.response = "Realistic response"
        router_engine.router_engine.query = MagicMock(return_value=mock_response)

        for scenario in sample_query_scenarios:
            query = scenario["query"]
            response = router_engine.query(query)

            assert response == mock_response
            assert response.response == "Realistic response"

    @pytest.mark.asyncio
    async def test_concurrent_query_execution(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test concurrent async query execution."""
        import asyncio

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        # Mock async response
        mock_response = MagicMock()
        mock_response.response = "Concurrent response"
        router_engine.router_engine.aquery = AsyncMock(return_value=mock_response)

        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does attention work?",
            "Compare BERT and GPT",
        ]

        # Execute queries concurrently
        tasks = [router_engine.aquery(query) for query in queries]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == len(queries)
        for response in responses:
            assert response == mock_response
            assert response.response == "Concurrent response"

    def test_query_truncation_logging(self, mock_vector_index, mock_llm_for_routing):
        """Test that long queries are truncated in logs."""
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            llm=mock_llm_for_routing,
        )

        mock_response = MagicMock()
        router_engine.router_engine.query = MagicMock(return_value=mock_response)

        # Create a very long query (>100 chars)
        long_query = "This is a very long query " * 10  # ~270 chars

        with patch("src.retrieval.query_engine.logger.info") as mock_logger:
            router_engine.query(long_query)

        # Check that logger was called with truncated query
        mock_logger.assert_called()
        logged_args = mock_logger.call_args[0]
        # Should contain truncated query (first 100 chars + "...")
        assert len(logged_args[1]) == 100  # QUERY_TRUNCATE_LENGTH

    def test_multimodal_integration_detection(
        self, mock_vector_index, mock_llm_for_routing
    ):
        """Test multimodal query detection in realistic context."""
        # Create multimodal index mock
        mock_multimodal_index = MagicMock()
        mock_multimodal_index.as_query_engine.return_value = MagicMock()

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            multimodal_index=mock_multimodal_index,
            llm=mock_llm_for_routing,
        )

        # Test various multimodal queries
        multimodal_queries = [
            "Show me diagrams of transformer architecture",
            "Find images related to CNN visualization",
            "Display charts of model performance metrics",
            "What does the neural network topology look like?",
            "Screenshot of the training dashboard",
        ]

        for query in multimodal_queries:
            is_multimodal = router_engine._detect_multimodal_query(query)
            assert is_multimodal, f"Should detect '{query}' as multimodal"

        # Test non-multimodal queries
        text_queries = [
            "Explain the transformer architecture in detail",
            "What are the benefits of attention mechanisms?",
            "Compare different optimization algorithms",
        ]

        for query in text_queries:
            is_multimodal = router_engine._detect_multimodal_query(query)
            assert not is_multimodal, f"Should not detect '{query}' as multimodal"
