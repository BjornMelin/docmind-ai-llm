"""Comprehensive unit tests for ToolFactory with all creation patterns.

This test suite provides systematic coverage of tool factory patterns including:
- Tool creation from different index types
- Mock LLM integration with predictable responses
- Error handling and fallback mechanisms
- Tool metadata validation and configuration
- Reranking integration patterns

Test Strategy:
- Mock boundaries (indexes, rerankers) not implementation
- Test behavior and configuration correctness
- Validate tool metadata and descriptions
- Test error recovery and fallback patterns

Markers:
- @pytest.mark.unit: Fast synchronous logic tests
- @pytest.mark.asyncio: Async tool behavior tests
"""

from unittest.mock import Mock, patch

import pytest
from llama_index.core.tools import QueryEngineTool
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from src.agents.tool_factory import ToolFactory
from src.config import settings


@pytest.mark.unit
class TestToolFactory:
    """Comprehensive test suite for ToolFactory tool creation patterns."""

    def test_create_query_tool_basic_configuration(self):
        """Test basic query tool creation with standard configuration."""
        # Given: Mock query engine
        mock_query_engine = Mock()
        tool_name = "test_search"
        tool_description = "Search test documents for information"

        # When: Creating query tool
        tool = ToolFactory.create_query_tool(
            query_engine=mock_query_engine, name=tool_name, description=tool_description
        )

        # Then: Tool is properly configured
        assert isinstance(tool, QueryEngineTool)
        assert tool.metadata.name == tool_name
        assert tool.metadata.description == tool_description
        assert tool.metadata.return_direct is False  # Allow agent reasoning

    def test_create_query_tool_metadata_validation(self):
        """Test tool creation with comprehensive metadata validation."""
        # Given: Mock query engine with complex description
        mock_query_engine = Mock()
        complex_description = (
            "Advanced hybrid search tool that combines dense and sparse embeddings "
            "with ColBERT reranking for optimal document retrieval and analysis"
        )

        # When: Creating tool with detailed metadata
        tool = ToolFactory.create_query_tool(
            query_engine=mock_query_engine,
            name="hybrid_search",
            description=complex_description,
        )

        # Then: Metadata is preserved correctly
        assert tool.metadata.name == "hybrid_search"
        assert len(tool.metadata.description) > 100  # Complex description preserved
        assert "hybrid" in tool.metadata.description.lower()
        assert "embeddings" in tool.metadata.description
        assert tool.metadata.return_direct is False

    def test_create_reranker_with_valid_model(self):
        """Test ColBERT reranker creation with configured model."""
        # Given: Settings with reranker model configured
        with patch.object(
            settings.retrieval, "reranker_model", "BAAI/bge-reranker-v2-m3"
        ):
            # When: Creating reranker
            reranker = ToolFactory._create_reranker()

            # Then: Reranker is created successfully
            assert reranker is not None
            assert isinstance(reranker, ColbertRerank)

    def test_create_reranker_without_model_returns_none(self):
        """Test reranker creation returns None when no model configured."""
        # Given: Settings without reranker model
        with patch.object(settings.retrieval, "reranker_model", ""):
            # When: Creating reranker
            reranker = ToolFactory._create_reranker()

            # Then: No reranker is created
            assert reranker is None

    def test_create_reranker_handles_initialization_error(self):
        """Test graceful handling of reranker initialization errors."""
        # Given: Settings with invalid reranker configuration
        with patch.object(settings.retrieval, "reranker_model", "invalid/model"):
            with patch(
                "llama_index.postprocessor.colbert_rerank.ColbertRerank",
                side_effect=Exception("Model not found"),
            ):
                # When: Creating reranker with error
                reranker = ToolFactory._create_reranker()

                # Then: Returns None on initialization error
                assert reranker is None

    @patch("src.agents.tool_factory.ToolFactory._create_reranker")
    def test_create_vector_search_tool(self, mock_create_reranker):
        """Test vector search tool creation with proper configuration."""
        # Given: Mock vector index and reranker
        mock_vector_index = Mock()
        mock_query_engine = Mock()
        mock_vector_index.as_query_engine.return_value = mock_query_engine

        mock_reranker = Mock(spec=ColbertRerank)
        mock_create_reranker.return_value = mock_reranker

        # When: Creating vector search tool
        tool = ToolFactory.create_vector_search_tool(mock_vector_index)

        # Then: Tool is created with proper configuration
        assert isinstance(tool, QueryEngineTool)
        assert "vector" in tool.metadata.name.lower()
        assert "semantic" in tool.metadata.description.lower()

        # Verify query engine creation called with reranker
        mock_vector_index.as_query_engine.assert_called_once()
        call_args = mock_vector_index.as_query_engine.call_args
        assert "node_postprocessors" in call_args[1]

    @patch("src.agents.tool_factory.ToolFactory._create_reranker")
    def test_create_vector_search_tool_without_reranker(self, mock_create_reranker):
        """Test vector search tool creation when no reranker available."""
        # Given: Mock vector index, no reranker
        mock_vector_index = Mock()
        mock_query_engine = Mock()
        mock_vector_index.as_query_engine.return_value = mock_query_engine
        mock_create_reranker.return_value = None

        # When: Creating vector search tool
        tool = ToolFactory.create_vector_search_tool(mock_vector_index)

        # Then: Tool is created without reranker
        assert isinstance(tool, QueryEngineTool)
        mock_vector_index.as_query_engine.assert_called_once()
        call_args = mock_vector_index.as_query_engine.call_args
        # Should still pass node_postprocessors as empty list
        assert "node_postprocessors" in call_args[1]

    def test_create_knowledge_graph_tool(self):
        """Test knowledge graph tool creation."""
        # Given: Mock knowledge graph index
        mock_kg_index = Mock()
        mock_query_engine = Mock()
        mock_kg_index.as_query_engine.return_value = mock_query_engine

        # When: Creating knowledge graph tool
        tool = ToolFactory.create_knowledge_graph_tool(mock_kg_index)

        # Then: Tool is created with KG-specific configuration
        assert isinstance(tool, QueryEngineTool)
        assert (
            "graph" in tool.metadata.name.lower()
            or "knowledge" in tool.metadata.name.lower()
        )
        assert "relationships" in tool.metadata.description.lower()

        # Verify proper similarity_top_k configuration
        mock_kg_index.as_query_engine.assert_called_once()
        call_args = mock_kg_index.as_query_engine.call_args
        assert "similarity_top_k" in call_args[1]
        assert call_args[1]["similarity_top_k"] == 10  # KG_SIMILARITY_TOP_K

    @patch("src.agents.tool_factory.ToolFactory._create_reranker")
    def test_create_hybrid_search_tool(self, mock_create_reranker):
        """Test hybrid search tool creation with retriever and reranker."""
        # Given: Mock retriever and reranker
        mock_retriever = Mock()
        mock_reranker = Mock(spec=ColbertRerank)
        mock_create_reranker.return_value = mock_reranker

        # When: Creating hybrid search tool
        tool = ToolFactory.create_hybrid_search_tool(mock_retriever)

        # Then: Tool is created with proper hybrid configuration
        assert isinstance(tool, QueryEngineTool)
        assert "hybrid" in tool.metadata.name.lower()
        assert "dense" in tool.metadata.description.lower()
        assert "sparse" in tool.metadata.description.lower()

        # Verify RetrieverQueryEngine creation with reranker
        mock_create_reranker.assert_called_once()

    @patch("src.agents.tool_factory.ToolFactory._create_reranker")
    def test_create_hybrid_search_tool_without_reranker(self, mock_create_reranker):
        """Test hybrid search tool creation when reranker unavailable."""
        # Given: Mock retriever, no reranker
        mock_retriever = Mock()
        mock_create_reranker.return_value = None

        # When: Creating hybrid search tool
        tool = ToolFactory.create_hybrid_search_tool(mock_retriever)

        # Then: Tool is still created without reranker
        assert isinstance(tool, QueryEngineTool)
        assert "hybrid" in tool.metadata.name.lower()

    def test_create_tools_from_indexes_comprehensive(self):
        """Test comprehensive tool creation from multiple index types."""
        # Given: Mock indexes and retriever
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        mock_kg_index = Mock()
        mock_kg_index.as_query_engine.return_value = Mock()

        mock_retriever = Mock()

        with patch(
            "src.agents.tool_factory.ToolFactory._create_reranker"
        ) as mock_create_reranker:
            mock_create_reranker.return_value = Mock(spec=ColbertRerank)

            # When: Creating tools from all index types
            tools = ToolFactory.create_tools_from_indexes(
                vector_index=mock_vector_index,
                kg_index=mock_kg_index,
                retriever=mock_retriever,
            )

        # Then: All tools are created
        assert len(tools) == 3  # vector, kg, hybrid
        tool_names = [tool.metadata.name for tool in tools]
        assert any("vector" in name.lower() for name in tool_names)
        assert any(
            "graph" in name.lower() or "knowledge" in name.lower()
            for name in tool_names
        )
        assert any("hybrid" in name.lower() for name in tool_names)

    def test_create_tools_from_indexes_partial(self):
        """Test tool creation with only some indexes available."""
        # Given: Only vector index available
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        with patch(
            "src.agents.tool_factory.ToolFactory._create_reranker"
        ) as mock_create_reranker:
            mock_create_reranker.return_value = None

            # When: Creating tools with partial indexes
            tools = ToolFactory.create_tools_from_indexes(
                vector_index=mock_vector_index, kg_index=None, retriever=None
            )

        # Then: Only available tools are created
        assert len(tools) == 1  # Only vector tool
        assert "vector" in tools[0].metadata.name.lower()

    def test_create_tools_from_indexes_empty(self):
        """Test tool creation with no indexes returns empty list."""
        # Given: No indexes available

        # When: Creating tools with no indexes
        tools = ToolFactory.create_tools_from_indexes(
            vector_index=None, kg_index=None, retriever=None
        )

        # Then: Empty tools list returned
        assert len(tools) == 0

    @patch("src.agents.tool_factory.logger")
    def test_error_handling_in_tool_creation(self, mock_logger):
        """Test error handling during tool creation."""
        # Given: Mock index that raises error
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.side_effect = Exception("Index error")

        # When: Creating tool with error
        with pytest.raises(Exception):
            ToolFactory.create_vector_search_tool(mock_vector_index)

    def test_tool_descriptions_contain_key_information(self):
        """Test that tool descriptions provide adequate information for agent decision-making."""
        # Given: Mock indexes
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        mock_kg_index = Mock()
        mock_kg_index.as_query_engine.return_value = Mock()

        mock_retriever = Mock()

        with patch(
            "src.agents.tool_factory.ToolFactory._create_reranker", return_value=None
        ):
            # When: Creating tools
            vector_tool = ToolFactory.create_vector_search_tool(mock_vector_index)
            kg_tool = ToolFactory.create_knowledge_graph_tool(mock_kg_index)
            hybrid_tool = ToolFactory.create_hybrid_search_tool(mock_retriever)

        # Then: Descriptions contain key decision-making information
        vector_desc = vector_tool.metadata.description.lower()
        assert any(
            keyword in vector_desc for keyword in ["semantic", "similar", "embedding"]
        )

        kg_desc = kg_tool.metadata.description.lower()
        assert any(
            keyword in kg_desc for keyword in ["relationship", "entities", "connection"]
        )

        hybrid_desc = hybrid_tool.metadata.description.lower()
        assert any(keyword in hybrid_desc for keyword in ["dense", "sparse", "fusion"])

    def test_settings_integration(self):
        """Test that ToolFactory properly integrates with settings configuration."""
        # Given: Check current settings integration

        # When: Accessing settings-based constants

        # Then: Settings are properly integrated
        assert settings.retrieval.reranking_top_k >= 1
        assert settings.retrieval.top_k >= 1

        # Verify constants match settings
        from src.agents.tool_factory import (
            DEFAULT_RERANKING_TOP_K,
            DEFAULT_VECTOR_SIMILARITY_TOP_K,
        )

        assert settings.retrieval.reranking_top_k == DEFAULT_RERANKING_TOP_K
        assert settings.retrieval.top_k == DEFAULT_VECTOR_SIMILARITY_TOP_K


@pytest.mark.asyncio
class TestToolFactoryAsync:
    """Async tests for ToolFactory behavior with async components."""

    async def test_async_tool_integration(self):
        """Test tool factory creates tools compatible with async workflows."""
        # Given: Mock async query engine
        mock_query_engine = Mock()
        mock_query_engine.aquery = Mock()

        # When: Creating tool for async usage
        tool = ToolFactory.create_query_tool(
            query_engine=mock_query_engine,
            name="async_search",
            description="Async search tool",
        )

        # Then: Tool is created and supports async operations
        assert isinstance(tool, QueryEngineTool)
        assert hasattr(tool.query_engine, "aquery") or hasattr(
            tool.query_engine, "query"
        )

    async def test_async_error_handling(self):
        """Test async error handling in tool operations."""
        # Given: Mock query engine that raises async errors
        mock_query_engine = Mock()
        mock_query_engine.query.side_effect = Exception("Async query error")

        tool = ToolFactory.create_query_tool(
            query_engine=mock_query_engine,
            name="error_tool",
            description="Tool that generates errors",
        )

        # When/Then: Tool creation succeeds even if async operations might fail
        assert isinstance(tool, QueryEngineTool)
        # Error handling should be managed at the agent level, not tool factory level
