"""Unit tests for ToolFactory covering edge cases and integrations.

Focus areas:
- Edge cases and error handling
- Configuration validation paths
- Integration with reranker and query engines via public methods only

Marker:
- @pytest.mark.unit: fast synchronous logic tests
"""

from unittest.mock import Mock, patch

import pytest
from llama_index.core.tools import QueryEngineTool
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from src.agents.tool_factory import KG_SIMILARITY_TOP_K, ToolFactory
from src.config import settings


@pytest.mark.unit
class TestToolFactoryEdgeCases:
    """Test edge cases and additional scenarios for ToolFactory."""

    def test_kg_similarity_top_k_constant(self):
        """Test KG_SIMILARITY_TOP_K constant value."""
        assert KG_SIMILARITY_TOP_K == 10
        assert isinstance(KG_SIMILARITY_TOP_K, int)

    def test_create_kg_search_tool_with_none_index(self):
        """Test knowledge graph tool creation with None index."""
        result = ToolFactory.create_kg_search_tool(None)
        assert result is None

    def test_create_kg_search_tool_with_false_index(self):
        """Test knowledge graph tool creation with falsy index."""
        result = ToolFactory.create_kg_search_tool(False)
        assert result is None

    def test_create_kg_search_tool_with_empty_dict(self):
        """Test knowledge graph tool creation with empty dict (falsy)."""
        result = ToolFactory.create_kg_search_tool({})
        assert result is None

    @patch("src.agents.tool_factory.ColbertRerank")
    def test_create_kg_search_tool_detailed_config(self, mock_colbert):
        """Test detailed configuration of knowledge graph tool."""
        # Given: Mock KG index and reranker
        mock_kg_index = Mock()
        mock_query_engine = Mock()
        mock_kg_index.as_query_engine.return_value = mock_query_engine

        mock_reranker = Mock(spec=ColbertRerank)
        mock_colbert.return_value = mock_reranker

        # When: Creating KG search tool
        with (
            patch.object(
                settings.retrieval,
                "reranker_model",
                "BAAI/bge-reranker-v2-m3",
            ),
            patch.object(settings.retrieval, "reranking_top_k", 10),
        ):
            tool = ToolFactory.create_kg_search_tool(mock_kg_index)

        # Then: Tool is configured with proper KG settings
        assert isinstance(tool, QueryEngineTool)

        # Verify as_query_engine was called with correct parameters
        # Verify call args without deep equality on postprocessors list
        _, kwargs = mock_kg_index.as_query_engine.call_args
        assert kwargs["similarity_top_k"] == KG_SIMILARITY_TOP_K
        assert kwargs["include_text"] is True
        assert kwargs["verbose"] is False
        assert isinstance(kwargs["node_postprocessors"], list)
        assert len(kwargs["node_postprocessors"]) == 1

    def test_create_kg_search_tool_no_reranker(self):
        """Test KG tool creation without reranker."""
        # Given: Mock KG index, no reranker
        mock_kg_index = Mock()
        mock_query_engine = Mock()
        mock_kg_index.as_query_engine.return_value = mock_query_engine

        # When: Creating KG search tool
        with patch.object(settings.retrieval, "reranker_model", ""):
            tool = ToolFactory.create_kg_search_tool(mock_kg_index)

        # Then: Tool is created without reranker
        assert isinstance(tool, QueryEngineTool)

        # Verify as_query_engine was called with empty postprocessors
        mock_kg_index.as_query_engine.assert_called_once_with(
            similarity_top_k=KG_SIMILARITY_TOP_K,
            include_text=True,
            node_postprocessors=[],
            verbose=False,
        )

    @patch("src.agents.tool_factory.ColbertRerank")
    def test_create_hybrid_vector_tool(self, mock_colbert):
        """Test hybrid vector tool creation as fallback."""
        # Given: Mock vector index and reranker
        mock_vector_index = Mock()
        mock_query_engine = Mock()
        mock_vector_index.as_query_engine.return_value = mock_query_engine

        mock_reranker = Mock(spec=ColbertRerank)
        mock_colbert.return_value = mock_reranker

        # When: Creating hybrid vector tool
        with (
            patch.object(
                settings.retrieval,
                "reranker_model",
                "BAAI/bge-reranker-v2-m3",
            ),
            patch.object(settings.retrieval, "reranking_top_k", 10),
        ):
            tool = ToolFactory.create_hybrid_vector_tool(mock_vector_index)

        # Then: Tool is created with hybrid configuration
        assert isinstance(tool, QueryEngineTool)
        assert "hybrid" in tool.metadata.name.lower()
        assert "vector" in tool.metadata.name.lower()
        assert "dense" in tool.metadata.description.lower()
        assert "sparse" in tool.metadata.description.lower()

        # Verify query engine creation
        _, kwargs = mock_vector_index.as_query_engine.call_args
        assert kwargs["similarity_top_k"] == settings.retrieval.top_k
        assert kwargs["verbose"] is False
        assert isinstance(kwargs["node_postprocessors"], list)
        assert len(kwargs["node_postprocessors"]) == 1

    def test_create_basic_tools_with_all_components(self):
        """Test create_basic_tools with comprehensive index data."""
        # Given: Mock all index types
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        mock_kg_index = Mock()
        mock_kg_index.as_query_engine.return_value = Mock()

        mock_retriever = Mock()

        index_data = {
            "vector": mock_vector_index,
            "kg": mock_kg_index,
            "retriever": mock_retriever,
        }

        with patch.object(settings.retrieval, "reranker_model", ""):
            # When: Creating basic tools
            tools = ToolFactory.create_basic_tools(index_data)

        # Then: All tools are created
        assert len(tools) == 3  # hybrid_search, kg_search, vector_search
        tool_names = [tool.metadata.name for tool in tools]

        # Check that we have the expected tool types
        assert any("hybrid" in name.lower() for name in tool_names)
        assert any(
            "graph" in name.lower() or "knowledge" in name.lower()
            for name in tool_names
        )
        assert any("vector" in name.lower() for name in tool_names)

    def test_create_basic_tools_with_partial_data(self):
        """Test create_basic_tools with partial index data."""
        # Given: Only vector index
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        index_data = {
            "vector": mock_vector_index,
            # Missing kg and retriever
        }

        with patch.object(settings.retrieval, "reranker_model", ""):
            # When: Creating basic tools
            tools = ToolFactory.create_basic_tools(index_data)

        # Then: Only vector tools are created (hybrid vector + vector search)
        assert len(tools) == 2  # hybrid_vector (fallback) + vector_search

    def test_create_basic_tools_with_empty_data(self):
        """Test create_basic_tools with empty index data."""
        # Given: Empty index data
        index_data = {}

        # When: Creating basic tools
        tools = ToolFactory.create_basic_tools(index_data)

        # Then: Empty list returned
        assert len(tools) == 0

    def test_create_basic_tools_with_none_values(self):
        """Test create_basic_tools with None values in data."""
        # Given: Index data with None values
        index_data = {
            "vector": None,
            "kg": None,
            "retriever": None,
        }

        # When: Creating basic tools
        tools = ToolFactory.create_basic_tools(index_data)

        # Then: Empty list returned
        assert len(tools) == 0

    @patch("src.agents.tool_factory.logger")
    def test_create_tools_from_indexes_logging(self, mock_logger):
        """Test logging behavior during tool creation."""
        # Given: Mock vector index only
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        with patch.object(settings.retrieval, "reranker_model", ""):
            # When: Creating tools
            ToolFactory.create_tools_from_indexes(
                vector_index=mock_vector_index,
                kg_index=None,
                retriever=None,
            )

        # Then: Appropriate log messages are generated
        # Check that info logging was called
        assert mock_logger.info.called

    @patch("src.agents.tool_factory.logger")
    def test_create_tools_from_indexes_no_vector_index(self, mock_logger):
        """Test tools creation with no vector index (error case)."""
        # Given: No vector index

        # When: Creating tools without vector index
        tools = ToolFactory.create_tools_from_indexes(
            vector_index=None,
            kg_index=Mock(),
            retriever=Mock(),
        )

        # Then: Empty list and error logged
        assert len(tools) == 0
        mock_logger.error.assert_called_once_with(
            "Vector index is required for tool creation"
        )

    @patch("src.agents.tool_factory.ColbertRerank")
    def test_reranker_integration_via_vector_tool(self, mock_colbert):
        """Test reranker integration by creating a vector tool (public path)."""
        mock_reranker = Mock(spec=ColbertRerank)
        mock_colbert.return_value = mock_reranker

        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        with (
            patch.object(settings.retrieval, "reranker_model", "custom/reranker"),
            patch.object(settings.retrieval, "reranking_top_k", 15),
        ):
            ToolFactory.create_vector_search_tool(mock_vector_index)

        mock_colbert.assert_called_once_with(
            top_n=15, model="custom/reranker", keep_retrieval_score=True
        )

    @patch("src.agents.tool_factory.ColbertRerank")
    def test_reranker_disabled_via_settings(self, mock_colbert):
        """When reranker is disabled by settings, no postprocessors are used."""
        mock_vector_index = Mock()
        mock_query_engine = Mock()
        mock_vector_index.as_query_engine.return_value = mock_query_engine

        with patch.object(settings.retrieval, "reranker_model", ""):
            ToolFactory.create_vector_search_tool(mock_vector_index)

        mock_colbert.assert_not_called()
        mock_vector_index.as_query_engine.assert_called_once()
        _, kwargs = mock_vector_index.as_query_engine.call_args
        assert kwargs.get("node_postprocessors") == []

    @patch("src.agents.tool_factory.ColbertRerank")
    def test_reranker_import_error_falls_back(self, mock_colbert):
        """Import error during ColbertRerank creation yields no postprocessor."""
        mock_colbert.side_effect = ImportError("Module not found")
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        with patch.object(settings.retrieval, "reranker_model", "test/model"):
            ToolFactory.create_vector_search_tool(mock_vector_index)

        _, kwargs = mock_vector_index.as_query_engine.call_args
        assert kwargs.get("node_postprocessors") == []

    @patch("src.agents.tool_factory.ColbertRerank")
    def test_reranker_value_error_falls_back(self, mock_colbert):
        """Value error during ColbertRerank creation yields no postprocessor."""
        mock_colbert.side_effect = ValueError("Invalid configuration")
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        with patch.object(settings.retrieval, "reranker_model", "test/model"):
            ToolFactory.create_vector_search_tool(mock_vector_index)

        _, kwargs = mock_vector_index.as_query_engine.call_args
        assert kwargs.get("node_postprocessors") == []

    @patch("src.agents.tool_factory.ColbertRerank")
    def test_reranker_attribute_error_falls_back(self, mock_colbert):
        """Attribute error during ColbertRerank creation yields no postprocessor."""
        mock_colbert.side_effect = AttributeError("Missing attribute")
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        with patch.object(settings.retrieval, "reranker_model", "test/model"):
            ToolFactory.create_vector_search_tool(mock_vector_index)

        _, kwargs = mock_vector_index.as_query_engine.call_args
        assert kwargs.get("node_postprocessors") == []

    def test_tool_metadata_completeness(self):
        """Test that tool metadata contains all required information."""
        # Given: Mock components
        mock_query_engine = Mock()

        # When: Creating a query tool
        tool = ToolFactory.create_query_tool(
            query_engine=mock_query_engine,
            name="test_tool",
            description="Test tool description",
        )

        # Then: Metadata is complete
        assert hasattr(tool, "metadata")
        assert hasattr(tool.metadata, "name")
        assert hasattr(tool.metadata, "description")
        assert hasattr(tool.metadata, "return_direct")
        assert tool.metadata.return_direct is False

    def test_settings_constants_validation(self):
        """Test validation of settings-based constants."""
        from src.agents.tool_factory import (
            DEFAULT_RERANKING_TOP_K,
            DEFAULT_VECTOR_SIMILARITY_TOP_K,
        )

        # Verify constants are positive integers
        assert isinstance(DEFAULT_RERANKING_TOP_K, int)
        assert isinstance(DEFAULT_VECTOR_SIMILARITY_TOP_K, int)
        assert DEFAULT_RERANKING_TOP_K > 0
        assert DEFAULT_VECTOR_SIMILARITY_TOP_K > 0

        # Verify they match current settings
        assert settings.retrieval.reranking_top_k == DEFAULT_RERANKING_TOP_K
        assert settings.retrieval.top_k == DEFAULT_VECTOR_SIMILARITY_TOP_K

    @patch("src.agents.tool_factory.RetrieverQueryEngine")
    def test_hybrid_search_tool_engine_creation(self, mock_retriever_engine):
        """Test that hybrid search tool creates RetrieverQueryEngine correctly."""
        # Given: Mock retriever and RetrieverQueryEngine
        mock_retriever = Mock()
        mock_engine_instance = Mock()
        mock_retriever_engine.return_value = mock_engine_instance

        with patch.object(settings.retrieval, "reranker_model", ""):
            # When: Creating hybrid search tool
            ToolFactory.create_hybrid_search_tool(mock_retriever)

            # Then: RetrieverQueryEngine was created correctly
            mock_retriever_engine.assert_called_once_with(
                retriever=mock_retriever,
                node_postprocessors=[],
            )

    def test_tool_descriptions_agent_guidance(self):
        """Test that tool descriptions provide clear guidance for agent selection."""
        # Given: Mock components
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        mock_kg_index = Mock()
        mock_kg_index.as_query_engine.return_value = Mock()

        mock_retriever = Mock()

        with patch.object(settings.retrieval, "reranker_model", ""):
            # When: Creating different tool types
            vector_tool = ToolFactory.create_vector_search_tool(mock_vector_index)
            kg_tool = ToolFactory.create_kg_search_tool(mock_kg_index)
            hybrid_tool = ToolFactory.create_hybrid_search_tool(mock_retriever)
            hybrid_vector_tool = ToolFactory.create_hybrid_vector_tool(
                mock_vector_index
            )

        # Then: Each tool has distinct, informative descriptions
        descriptions = [
            vector_tool.metadata.description,
            kg_tool.metadata.description,
            hybrid_tool.metadata.description,
            hybrid_vector_tool.metadata.description,
        ]

        # All descriptions should be substantial
        for desc in descriptions:
            assert len(desc) > 50  # Substantial description
            assert "." in desc  # Proper sentences

        # Vector tool should mention semantic similarity
        vector_desc = vector_tool.metadata.description.lower()
        assert any(
            word in vector_desc for word in ["semantic", "similarity", "embeddings"]
        )

        # KG tool should mention relationships
        kg_desc = kg_tool.metadata.description.lower()
        assert any(
            word in kg_desc for word in ["relationship", "entities", "connections"]
        )

        # Hybrid tools should mention fusion or combining approaches
        hybrid_desc = hybrid_tool.metadata.description.lower()
        hybrid_vector_desc = hybrid_vector_tool.metadata.description.lower()
        assert any(
            word in hybrid_desc for word in ["fusion", "combines", "dense", "sparse"]
        )
        assert any(
            word in hybrid_vector_desc
            for word in ["hybrid", "combines", "dense", "sparse"]
        )


@pytest.mark.unit
class TestToolFactoryIntegration:
    """Integration tests for ToolFactory with various component combinations."""

    @patch("src.agents.tool_factory.ColbertRerank")
    def test_full_pipeline_with_all_components(self, mock_colbert):
        """Test full tool creation pipeline with all components available."""
        # Given: All components available
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        mock_kg_index = Mock()
        mock_kg_index.as_query_engine.return_value = Mock()

        mock_retriever = Mock()
        mock_reranker = Mock(spec=ColbertRerank)
        mock_colbert.return_value = mock_reranker

        # When: Creating tools from all indexes
        with (
            patch.object(
                settings.retrieval, "reranker_model", "BAAI/bge-reranker-v2-m3"
            ),
            patch.object(settings.retrieval, "reranking_top_k", 10),
        ):
            tools = ToolFactory.create_tools_from_indexes(
                vector_index=mock_vector_index,
                kg_index=mock_kg_index,
                retriever=mock_retriever,
            )

        # Then: All tools created with reranker integration
        assert len(tools) == 3

        # Verify reranker was used (Colbert constructed)
        assert mock_colbert.call_count >= 3

        # Verify all query engines were created
        assert mock_vector_index.as_query_engine.call_count >= 1
        assert mock_kg_index.as_query_engine.call_count == 1

    def test_degraded_pipeline_no_retriever(self):
        """Test tool creation pipeline when retriever is unavailable."""
        # Given: No retriever available
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        mock_kg_index = Mock()
        mock_kg_index.as_query_engine.return_value = Mock()

        # When: Creating tools without retriever
        with patch.object(settings.retrieval, "reranker_model", ""):
            tools = ToolFactory.create_tools_from_indexes(
                vector_index=mock_vector_index,
                kg_index=mock_kg_index,
                retriever=None,
            )

        # Then: Fallback tools are created
        assert len(tools) == 3  # hybrid_vector (fallback), kg, vector
        tool_names = [tool.metadata.name for tool in tools]

        # Should have hybrid vector tool as fallback
        assert any(
            "hybrid" in name.lower() and "vector" in name.lower() for name in tool_names
        )

    def test_minimal_pipeline_vector_only(self):
        """Test minimal tool creation with only vector index."""
        # Given: Only vector index
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        with patch.object(settings.retrieval, "reranker_model", ""):
            # When: Creating tools with minimal setup
            tools = ToolFactory.create_tools_from_indexes(
                vector_index=mock_vector_index,
                kg_index=None,
                retriever=None,
            )

        # Then: Basic tools are created
        assert len(tools) == 2  # hybrid_vector (fallback) + vector

        # Verify vector index was used
        assert mock_vector_index.as_query_engine.call_count >= 1

    def test_error_recovery_in_tool_creation(self):
        """Test error recovery during tool creation."""
        # Given: Vector index that works, KG index that fails
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        mock_kg_index = Mock()
        mock_kg_index.as_query_engine.side_effect = Exception("KG creation failed")

        with patch.object(settings.retrieval, "reranker_model", ""):
            # When: Creating tools with partial failures
            # The KG tool creation will fail, but others should succeed
            # Test that vector tool creation succeeds
            vector_tool = ToolFactory.create_vector_search_tool(mock_vector_index)
            assert isinstance(vector_tool, QueryEngineTool)

            # Test that KG tool creation fails with expected exception
            with pytest.raises(Exception, match="KG creation failed"):
                ToolFactory.create_kg_search_tool(mock_kg_index)
