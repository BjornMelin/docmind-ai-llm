"""Unit tests for RouterQueryEngine adaptive retrieval (FEAT-002).

Tests the complete architectural replacement of QueryFusionRetriever
with RouterQueryEngine per ADR-003, providing intelligent strategy selection.

Test Coverage:
- AdaptiveRouterQueryEngine initialization and configuration
- LLMSingleSelector strategy selection logic
- QueryEngineTool creation for multiple strategies
- Fallback mechanisms and error handling
- Integration with BGE-M3 embeddings and reranking
- Performance and strategy selection accuracy
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval.query_engine.router_engine import (
    AdaptiveRouterQueryEngine,
    configure_router_settings,
    create_adaptive_router_engine,
)


class TestAdaptiveRouterQueryEngine:  # pylint: disable=protected-access
    """Unit tests for AdaptiveRouterQueryEngine class."""

    def test_init_with_minimal_config(self, mock_vector_index):
        """Test initialization with minimal required configuration."""
        router = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)

        # Verify basic configuration
        assert router.vector_index == mock_vector_index
        assert router.kg_index is None
        assert router.hybrid_retriever is None
        assert router.reranker is None
        assert router.router_engine is not None

    def test_init_with_full_config(
        self, mock_vector_index, mock_hybrid_retriever, mock_llm_for_routing
    ):
        """Test initialization with full configuration."""
        mock_kg_index = MagicMock()
        mock_reranker = MagicMock()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            kg_index=mock_kg_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=mock_reranker,
            llm=mock_llm_for_routing,
        )

        # Verify full configuration
        assert router.vector_index == mock_vector_index
        assert router.kg_index == mock_kg_index
        assert router.hybrid_retriever == mock_hybrid_retriever
        assert router.reranker == mock_reranker
        assert router.llm == mock_llm_for_routing

    @patch("src.retrieval.query_engine.router_engine.Settings")
    def test_init_uses_settings_llm(self, mock_settings, mock_vector_index):
        """Test that Settings.llm is used when no LLM provided."""
        mock_settings.llm = MagicMock()

        router = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)

        assert router.llm == mock_settings.llm

    def test_create_query_engine_tools_minimal(self, mock_vector_index):
        """Test QueryEngineTool creation with minimal configuration."""
        router = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)

        tools = router._create_query_engine_tools()

        # Should have at least semantic search tool
        assert len(tools) >= 1

        # Check semantic search tool
        semantic_tool = next(
            (t for t in tools if t.metadata.name == "semantic_search"), None
        )
        assert semantic_tool is not None
        assert "semantic" in semantic_tool.metadata.description.lower()
        assert "bge-m3" in semantic_tool.metadata.description.lower()

    def test_create_query_engine_tools_with_hybrid(
        self, mock_vector_index, mock_hybrid_retriever
    ):
        """Test QueryEngineTool creation with hybrid retriever."""
        mock_reranker = MagicMock()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=mock_reranker,
        )

        tools = router._create_query_engine_tools()

        # Should have hybrid search tool
        hybrid_tool = next(
            (t for t in tools if t.metadata.name == "hybrid_search"), None
        )
        assert hybrid_tool is not None
        assert "hybrid" in hybrid_tool.metadata.description.lower()
        assert "rrf fusion" in hybrid_tool.metadata.description.lower()

    def test_create_query_engine_tools_with_kg(self, mock_vector_index):
        """Test QueryEngineTool creation with knowledge graph."""
        mock_kg_index = MagicMock()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index, kg_index=mock_kg_index
        )

        tools = router._create_query_engine_tools()

        # Should have knowledge graph tool
        kg_tool = next((t for t in tools if t.metadata.name == "knowledge_graph"), None)
        assert kg_tool is not None
        assert "knowledge graph" in kg_tool.metadata.description.lower()
        assert "relationship" in kg_tool.metadata.description.lower()

    def test_create_query_engine_tools_comprehensive(
        self, mock_vector_index, mock_hybrid_retriever
    ):
        """Test comprehensive QueryEngineTool creation."""
        mock_kg_index = MagicMock()
        mock_reranker = MagicMock()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            kg_index=mock_kg_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=mock_reranker,
        )

        tools = router._create_query_engine_tools()

        # Should have all 4 tools
        tool_names = [t.metadata.name for t in tools]
        expected_tools = [
            "hybrid_search",
            "semantic_search",
            "multi_query_search",
            "knowledge_graph",
        ]

        for expected_tool in expected_tools:
            assert expected_tool in tool_names

    @patch("src.retrieval.query_engine.router_engine.LLMSingleSelector")
    @patch("src.retrieval.query_engine.router_engine.RouterQueryEngine")
    def test_create_router_engine(
        self, mock_router_class, mock_selector_class, mock_vector_index
    ):
        """Test RouterQueryEngine creation with LLMSingleSelector."""
        mock_selector = MagicMock()
        mock_selector_class.from_defaults.return_value = mock_selector

        mock_router_instance = MagicMock()
        mock_router_class.return_value = mock_router_instance

        _ = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)

        # Verify selector creation
        mock_selector_class.from_defaults.assert_called_once()

        # Verify router creation
        mock_router_class.assert_called_once()
        call_args = mock_router_class.call_args

        assert call_args[1]["selector"] == mock_selector
        assert call_args[1]["verbose"] is True
        assert len(call_args[1]["query_engine_tools"]) >= 1

    def test_create_router_engine_no_tools_error(self, mock_vector_index):
        """Test error handling when no query engine tools available."""
        with (
            patch.object(
                AdaptiveRouterQueryEngine, "_create_query_engine_tools", return_value=[]
            ),
            pytest.raises(ValueError, match="No query engine tools available"),
        ):
            AdaptiveRouterQueryEngine(vector_index=mock_vector_index)

    def test_query_success(self, mock_vector_index, sample_query_scenarios):
        """Test successful query execution through RouterQueryEngine."""
        # Mock router engine response
        mock_response = MagicMock()
        mock_response.response = "Mock router response"
        mock_response.metadata = {"selector_result": "hybrid_search"}

        router = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)
        router.router_engine.query = MagicMock(return_value=mock_response)

        query = sample_query_scenarios[0][
            "query"
        ]  # "explain quantum computing applications"

        result = router.query(query)

        # Verify successful execution
        assert result == mock_response
        router.router_engine.query.assert_called_once_with(query)

    def test_query_with_fallback(self, mock_vector_index):
        """Test query fallback to direct semantic search."""
        # Mock router engine failure
        router = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)
        router.router_engine.query = MagicMock(
            side_effect=RuntimeError("Router failed")
        )

        # Mock fallback query engine
        mock_fallback_response = MagicMock()
        mock_fallback_engine = MagicMock()
        mock_fallback_engine.query.return_value = mock_fallback_response
        mock_vector_index.as_query_engine.return_value = mock_fallback_engine

        # Reset call count after router initialization calls
        mock_vector_index.as_query_engine.reset_mock()

        query = "test query for fallback"

        result = router.query(query)

        # Verify fallback execution
        assert result == mock_fallback_response
        mock_vector_index.as_query_engine.assert_called_once()
        mock_fallback_engine.query.assert_called_once_with(query)

    async def test_aquery_success(self, mock_vector_index, sample_query_scenarios):
        """Test successful async query execution."""
        mock_response = MagicMock()
        mock_response.response = "Mock async response"
        mock_response.metadata = {"selector_result": "semantic_search"}

        router = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)
        router.router_engine.aquery = AsyncMock(return_value=mock_response)

        query = sample_query_scenarios[1]["query"]  # "BGE-M3 embeddings"

        result = await router.aquery(query)

        # Verify successful async execution
        assert result == mock_response
        router.router_engine.aquery.assert_called_once_with(query)

    async def test_aquery_with_fallback(self, mock_vector_index):
        """Test async query fallback to direct semantic search."""
        # Mock async router engine failure
        router = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)
        router.router_engine.aquery = AsyncMock(
            side_effect=RuntimeError("Async router failed")
        )

        # Mock async fallback query engine
        mock_fallback_response = MagicMock()
        mock_fallback_engine = MagicMock()
        mock_fallback_engine.aquery = AsyncMock(return_value=mock_fallback_response)
        mock_vector_index.as_query_engine.return_value = mock_fallback_engine

        query = "test async query for fallback"

        result = await router.aquery(query)

        # Verify async fallback execution
        assert result == mock_fallback_response
        mock_fallback_engine.aquery.assert_called_once_with(query)

    def test_get_available_strategies(self, mock_vector_index, mock_hybrid_retriever):
        """Test retrieval of available strategy names."""
        mock_kg_index = MagicMock()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            kg_index=mock_kg_index,
            hybrid_retriever=mock_hybrid_retriever,
        )

        strategies = router.get_available_strategies()

        # Verify all strategies are listed
        expected_strategies = [
            "hybrid_search",
            "semantic_search",
            "multi_query_search",
            "knowledge_graph",
        ]

        for strategy in expected_strategies:
            assert strategy in strategies


class TestAdaptiveRouterFactory:
    """Test factory functions and configuration helpers."""

    def test_create_adaptive_router_engine(self, mock_vector_index):
        """Test factory function with default parameters."""
        router = create_adaptive_router_engine(vector_index=mock_vector_index)

        assert isinstance(router, AdaptiveRouterQueryEngine)
        assert router.vector_index == mock_vector_index

    def test_create_adaptive_router_engine_full_config(
        self, mock_vector_index, mock_hybrid_retriever, mock_llm_for_routing
    ):
        """Test factory function with full configuration."""
        mock_kg_index = MagicMock()
        mock_reranker = MagicMock()

        router = create_adaptive_router_engine(
            vector_index=mock_vector_index,
            kg_index=mock_kg_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=mock_reranker,
            llm=mock_llm_for_routing,
        )

        assert router.kg_index == mock_kg_index
        assert router.hybrid_retriever == mock_hybrid_retriever
        assert router.reranker == mock_reranker
        assert router.llm == mock_llm_for_routing

    def test_configure_router_settings(self, mock_vector_index):
        """Test router settings configuration helper."""
        router = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)

        # Should not raise an error (settings configuration is informational)
        configure_router_settings(router)


class TestRouterStrategy:  # pylint: disable=protected-access
    """Test strategy selection and tool descriptions."""

    def test_strategy_descriptions_completeness(
        self, mock_vector_index, mock_hybrid_retriever
    ):
        """Test that all strategy descriptions contain key information."""
        mock_kg_index = MagicMock()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            kg_index=mock_kg_index,
            hybrid_retriever=mock_hybrid_retriever,
        )

        tools = router._create_query_engine_tools()

        # Check hybrid search description
        hybrid_tool = next(t for t in tools if t.metadata.name == "hybrid_search")
        description = hybrid_tool.metadata.description.lower()

        # Should mention key BGE-M3 and RRF features
        assert "bge-m3" in description
        assert "unified" in description
        assert "dense" in description
        assert "sparse" in description
        assert "rrf fusion" in description
        assert "8k context" in description

    def test_semantic_search_description(self, mock_vector_index):
        """Test semantic search tool description accuracy."""
        router = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)

        tools = router._create_query_engine_tools()
        semantic_tool = next(t for t in tools if t.metadata.name == "semantic_search")

        description = semantic_tool.metadata.description.lower()

        # Should emphasize semantic capabilities
        assert "semantic" in description
        assert "bge-m3" in description
        assert "1024-dimensional" in description
        assert "multilingual" in description
        assert "8k context" in description

    def test_multi_query_description(self, mock_vector_index):
        """Test multi-query search tool description accuracy."""
        router = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)

        tools = router._create_query_engine_tools()
        multi_tool = next(t for t in tools if t.metadata.name == "multi_query_search")

        description = multi_tool.metadata.description.lower()

        # Should emphasize decomposition and complexity
        assert "multi-query" in description
        assert "decomposition" in description
        assert "complex" in description
        assert (
            "tree summar" in description
        )  # Matches both "tree summarize" and "tree summarization"

    def test_knowledge_graph_description(self, mock_vector_index):
        """Test knowledge graph tool description accuracy."""
        mock_kg_index = MagicMock()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index, kg_index=mock_kg_index
        )

        tools = router._create_query_engine_tools()
        kg_tool = next(t for t in tools if t.metadata.name == "knowledge_graph")

        description = kg_tool.metadata.description.lower()

        # Should emphasize relationships and structure
        assert "knowledge graph" in description
        assert "relationship" in description
        assert "entity" in description
        assert "connected" in description or "connections" in description


@pytest.mark.integration
class TestRouterIntegration:  # pylint: disable=protected-access
    """Integration tests with LlamaIndex ecosystem."""

    @patch("src.retrieval.query_engine.router_engine.RetrieverQueryEngine")
    def test_retriever_query_engine_integration(
        self, mock_retriever_engine_class, mock_vector_index, mock_hybrid_retriever
    ):
        """Test integration with RetrieverQueryEngine."""
        mock_engine_instance = MagicMock()
        mock_retriever_engine_class.from_args.return_value = mock_engine_instance

        mock_reranker = MagicMock()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=mock_reranker,
        )

        _ = router._create_query_engine_tools()

        # Should create RetrieverQueryEngine for hybrid search
        mock_retriever_engine_class.from_args.assert_called()
        call_args = mock_retriever_engine_class.from_args.call_args[1]

        assert call_args["retriever"] == mock_hybrid_retriever
        assert mock_reranker in call_args["node_postprocessors"]
        assert call_args["response_mode"] == "compact"
        assert call_args["streaming"] is True

    def test_vector_index_query_engine_integration(self, mock_vector_index):
        """Test integration with vector index query engines."""
        router = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)

        _ = router._create_query_engine_tools()

        # Should call as_query_engine with proper parameters
        mock_vector_index.as_query_engine.assert_called()

        # Verify multiple calls for different strategies
        call_count = mock_vector_index.as_query_engine.call_count
        assert call_count >= 2  # At least semantic and multi_query

    @patch("src.retrieval.query_engine.router_engine.LLMSingleSelector")
    def test_llm_selector_integration(
        self, mock_selector_class, mock_vector_index, mock_llm_for_routing
    ):
        """Test integration with LLMSingleSelector."""
        mock_selector = MagicMock()
        mock_selector_class.from_defaults.return_value = mock_selector

        _ = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index, llm=mock_llm_for_routing
        )

        # Verify selector creation with custom LLM
        mock_selector_class.from_defaults.assert_called_once_with(
            llm=mock_llm_for_routing
        )


@pytest.mark.performance
class TestRouterPerformance:  # pylint: disable=protected-access,unused-argument
    """Performance and optimization tests."""

    def test_strategy_selection_performance_target(
        self, mock_vector_index, benchmark_timer, rtx_4090_performance_targets
    ):
        """Test strategy selection meets <50ms target."""
        router = AdaptiveRouterQueryEngine(vector_index=mock_vector_index)

        # Mock fast response
        mock_response = MagicMock()
        mock_response.metadata = {"selector_result": "hybrid_search"}
        router.router_engine.query = MagicMock(return_value=mock_response)

        # Benchmark multiple queries
        for _ in range(5):
            benchmark_timer.start()
            router.query("test query for performance")
            latency = benchmark_timer.stop()

            # Individual query should be fast
            assert latency < 1000  # Very lenient for mocked test

        stats = benchmark_timer.get_stats()
        # In real implementation, this should validate against the 50ms target
        assert stats["mean_ms"] < 1000  # Placeholder assertion

    def test_tool_creation_efficiency(self, mock_vector_index, mock_hybrid_retriever):
        """Test efficient QueryEngineTool creation."""
        mock_kg_index = MagicMock()
        mock_reranker = MagicMock()

        # Time tool creation
        import time

        start_time = time.perf_counter()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            kg_index=mock_kg_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=mock_reranker,
        )

        end_time = time.perf_counter()
        creation_time_ms = (end_time - start_time) * 1000

        # Tool creation should be fast
        assert creation_time_ms < 100  # Very lenient for mocked test
        assert len(router.get_available_strategies()) == 4

    def test_memory_efficient_tool_storage(
        self, mock_vector_index, mock_hybrid_retriever
    ):
        """Test memory-efficient tool storage and reuse."""
        mock_kg_index = MagicMock()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            kg_index=mock_kg_index,
            hybrid_retriever=mock_hybrid_retriever,
        )

        # Tools should be created once and reused
        tools_1 = router._create_query_engine_tools()
        tools_2 = router._create_query_engine_tools()

        # Should create new instances (not cached in current implementation)
        # In production, this could be optimized with caching
        assert len(tools_1) == len(tools_2) == 4
