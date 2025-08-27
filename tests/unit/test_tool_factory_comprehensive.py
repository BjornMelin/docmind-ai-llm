"""Comprehensive test coverage for tool_factory.py.

This test suite provides comprehensive coverage for the ToolFactory class,
focusing on business-critical tool creation, configuration management,
reranking integration, and error handling scenarios to achieve 70%+ coverage.

Key areas covered:
- All ToolFactory methods and static methods
- Configuration validation and edge cases
- ColBERT reranking integration
- Error handling and fallback mechanisms
- Tool metadata and descriptions
- Performance optimization scenarios
"""

from unittest.mock import MagicMock, call, patch

import pytest
from llama_index.core.tools import QueryEngineTool

# Import the module under test
from src.agents.tool_factory import ToolFactory
from src.config.settings import DocMindSettings


class TestToolFactoryBasicMethods:
    """Test basic ToolFactory methods."""

    def test_create_query_tool_basic(self):
        """Test basic query tool creation."""
        mock_query_engine = MagicMock()
        tool_name = "test_tool"
        tool_description = "Test tool description"

        result = ToolFactory.create_query_tool(
            mock_query_engine, tool_name, tool_description
        )

        assert isinstance(result, QueryEngineTool)
        assert result.query_engine == mock_query_engine
        assert result.metadata.name == tool_name
        assert result.metadata.description == tool_description
        assert result.metadata.return_direct is False

    def test_create_query_tool_with_empty_name(self):
        """Test query tool creation with empty name."""
        mock_query_engine = MagicMock()

        result = ToolFactory.create_query_tool(mock_query_engine, "", "Description")

        # Should still create tool even with empty name
        assert isinstance(result, QueryEngineTool)
        assert result.metadata.name == ""

    def test_create_query_tool_with_none_query_engine(self):
        """Test query tool creation with None query engine."""
        result = ToolFactory.create_query_tool(None, "test_tool", "Description")

        # Should create tool but with None query engine
        assert isinstance(result, QueryEngineTool)
        assert result.query_engine is None

    def test_create_query_tool_with_long_description(self):
        """Test query tool creation with very long description."""
        mock_query_engine = MagicMock()
        long_description = "Very long description. " * 100

        result = ToolFactory.create_query_tool(
            mock_query_engine, "test_tool", long_description
        )

        # Should handle long descriptions without issue
        assert result.metadata.description == long_description


class TestToolFactoryReranker:
    """Test ColBERT reranker creation and configuration."""

    def test_create_reranker_success(self):
        """Test successful reranker creation."""
        test_settings = DocMindSettings(
            reranker_model="colbert-ir/colbertv2.0", reranking_top_k=5
        )

        with (
            patch("src.agents.tool_factory.app_settings", test_settings),
            patch("src.agents.tool_factory.ColbertRerank") as mock_colbert_class,
        ):
            mock_reranker = MagicMock()
            mock_colbert_class.return_value = mock_reranker

            result = ToolFactory._create_reranker()

            assert result == mock_reranker
            mock_colbert_class.assert_called_once_with(
                top_n=5, model="colbert-ir/colbertv2.0", keep_retrieval_score=True
            )

    def test_create_reranker_no_model_configured(self):
        """Test reranker creation when no model is configured."""
        test_settings = DocMindSettings(reranker_model=None)

        with patch("agents.tool_factory.settings", test_settings):
            result = ToolFactory._create_reranker()

            assert result is None

    def test_create_reranker_empty_model_string(self):
        """Test reranker creation with empty model string."""
        test_settings = DocMindSettings(reranker_model="")

        with patch("agents.tool_factory.settings", test_settings):
            result = ToolFactory._create_reranker()

            assert result is None

    def test_create_reranker_default_top_k(self):
        """Test reranker creation with default top_k when not configured."""
        test_settings = DocMindSettings(
            reranker_model="colbert-ir/colbertv2.0", reranking_top_k=None
        )

        with (
            patch("src.agents.tool_factory.app_settings", test_settings),
            patch("src.agents.tool_factory.ColbertRerank") as mock_colbert_class,
        ):
            mock_reranker = MagicMock()
            mock_colbert_class.return_value = mock_reranker

            ToolFactory._create_reranker()

            # Should use default top_k of 5
            mock_colbert_class.assert_called_once_with(
                top_n=5, model="colbert-ir/colbertv2.0", keep_retrieval_score=True
            )

    def test_create_reranker_exception_handling(self):
        """Test reranker creation exception handling."""
        test_settings = DocMindSettings(
            reranker_model="invalid-model",
            reranking_top_k=5,
        )

        with (
            patch("src.agents.tool_factory.app_settings", test_settings),
            patch(
                "agents.tool_factory.ColbertRerank",
                side_effect=RuntimeError("Model not found"),
            ),
            patch("agents.tool_factory.logging.warning") as mock_warning,
        ):
            result = ToolFactory._create_reranker()

            assert result is None
            mock_warning.assert_called_once()

    def test_create_reranker_import_error_handling(self):
        """Test reranker creation when ColbertRerank is not available."""
        test_settings = DocMindSettings(
            reranker_model="colbert-ir/colbertv2.0", reranking_top_k=5
        )

        with (
            patch("src.agents.tool_factory.app_settings", test_settings),
            patch(
                "agents.tool_factory.ColbertRerank",
                side_effect=ImportError("ColbertRerank not installed"),
            ),
            patch("agents.tool_factory.logging.warning") as mock_warning,
        ):
            result = ToolFactory._create_reranker()

            assert result is None
            mock_warning.assert_called_once()


class TestToolFactoryVectorSearch:
    """Test vector search tool creation."""

    def test_create_vector_search_tool_success(self):
        """Test successful vector search tool creation."""
        mock_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine

        test_settings = DocMindSettings(top_k=5, debug_mode=False, reranker_model=None)

        with patch("agents.tool_factory.settings", test_settings):
            result = ToolFactory.create_vector_search_tool(mock_index)

            assert isinstance(result, QueryEngineTool)
            assert result.query_engine == mock_query_engine
            assert result.metadata.name == "vector_search"
            assert "semantic similarity search" in result.metadata.description.lower()

            # Verify query engine configuration
            mock_index.as_query_engine.assert_called_once_with(
                top_k=5, node_postprocessors=[], verbose=False
            )

    def test_create_vector_search_tool_with_reranker(self):
        """Test vector search tool creation with ColBERT reranker."""
        mock_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine

        test_settings = DocMindSettings(
            top_k=10,
            debug_mode=True,
            reranker_model="colbert-ir/colbertv2.0",
            reranking_top_k=3,
        )

        with (
            patch("src.agents.tool_factory.app_settings", test_settings),
            patch("src.agents.tool_factory.ColbertRerank") as mock_colbert_class,
        ):
            mock_reranker = MagicMock()
            mock_colbert_class.return_value = mock_reranker

            result = ToolFactory.create_vector_search_tool(mock_index)

            assert isinstance(result, QueryEngineTool)

            # Verify reranker was created and used
            mock_colbert_class.assert_called_once()
            mock_index.as_query_engine.assert_called_once_with(
                top_k=10,
                node_postprocessors=[mock_reranker],
                verbose=True,
            )

    def test_create_vector_search_tool_none_index(self):
        """Test vector search tool creation with None index."""
        with pytest.raises(AttributeError):
            ToolFactory.create_vector_search_tool(None)

    def test_create_vector_search_tool_default_settings(self):
        """Test vector search tool creation with default settings."""
        mock_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine

        test_settings = DocMindSettings(
            similarity_top_k=None,  # Should use default
            debug_mode=False,
            reranker_model=None,
        )

        with patch("agents.tool_factory.settings", test_settings):
            ToolFactory.create_vector_search_tool(mock_index)

            # Should use default similarity_top_k of 5
            mock_index.as_query_engine.assert_called_once_with(
                top_k=5, node_postprocessors=[], verbose=False
            )


class TestToolFactoryKnowledgeGraph:
    """Test knowledge graph search tool creation."""

    def test_create_kg_search_tool_success(self):
        """Test successful KG search tool creation."""
        mock_kg_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_kg_index.as_query_engine.return_value = mock_query_engine

        test_settings = DocMindSettings(debug_mode=False, reranker_model=None)

        with patch("agents.tool_factory.settings", test_settings):
            result = ToolFactory.create_kg_search_tool(mock_kg_index)

            assert isinstance(result, QueryEngineTool)
            assert result.query_engine == mock_query_engine
            assert result.metadata.name == "knowledge_graph"
            assert "knowledge graph search" in result.metadata.description.lower()
            assert "entity" in result.metadata.description.lower()

            # Verify KG-specific configuration
            mock_kg_index.as_query_engine.assert_called_once_with(
                top_k=10,  # KG uses higher top_k
                include_text=True,
                node_postprocessors=[],
                verbose=False,
            )

    def test_create_kg_search_tool_with_reranker(self):
        """Test KG search tool creation with reranker."""
        mock_kg_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_kg_index.as_query_engine.return_value = mock_query_engine

        test_settings = DocMindSettings(
            debug_mode=True, reranker_model="colbert-ir/colbertv2.0"
        )

        with (
            patch("src.agents.tool_factory.app_settings", test_settings),
            patch("src.agents.tool_factory.ColbertRerank") as mock_colbert_class,
        ):
            mock_reranker = MagicMock()
            mock_colbert_class.return_value = mock_reranker

            ToolFactory.create_kg_search_tool(mock_kg_index)

            # Verify reranker was used
            mock_kg_index.as_query_engine.assert_called_once_with(
                top_k=10,
                include_text=True,
                node_postprocessors=[mock_reranker],
                verbose=True,
            )

    def test_create_kg_search_tool_none_index(self):
        """Test KG search tool creation with None index."""
        result = ToolFactory.create_kg_search_tool(None)

        assert result is None

    def test_create_kg_search_tool_false_index(self):
        """Test KG search tool creation with falsy index."""
        result = ToolFactory.create_kg_search_tool(False)

        assert result is None

    def test_create_kg_search_tool_empty_index(self):
        """Test KG search tool creation with empty object."""
        # Test with an empty dict or similar falsy object
        result = ToolFactory.create_kg_search_tool({})

        assert result is None


class TestToolFactoryHybridSearch:
    """Test hybrid search tool creation."""

    def test_create_hybrid_search_tool_success(self):
        """Test successful hybrid search tool creation."""
        mock_retriever = MagicMock()

        with patch("agents.tool_factory.RetrieverQueryEngine") as mock_engine_class:
            mock_query_engine = MagicMock()
            mock_engine_class.return_value = mock_query_engine

            test_settings = DocMindSettings(reranker_model=None)

            with patch("agents.tool_factory.settings", test_settings):
                result = ToolFactory.create_hybrid_search_tool(mock_retriever)

                assert isinstance(result, QueryEngineTool)
                assert result.query_engine == mock_query_engine
                assert result.metadata.name == "hybrid_fusion_search"
                assert "rrf" in result.metadata.description.lower()
                assert "reciprocal rank fusion" in result.metadata.description.lower()

                # Verify engine configuration
                mock_engine_class.assert_called_once_with(
                    retriever=mock_retriever, node_postprocessors=[]
                )

    def test_create_hybrid_search_tool_with_reranker(self):
        """Test hybrid search tool creation with ColBERT reranker."""
        mock_retriever = MagicMock()

        with patch("agents.tool_factory.RetrieverQueryEngine") as mock_engine_class:
            mock_query_engine = MagicMock()
            mock_engine_class.return_value = mock_query_engine

            test_settings = DocMindSettings(
                reranker_model="colbert-ir/colbertv2.0", reranking_top_k=5
            )

            with (
                patch("src.agents.tool_factory.app_settings", test_settings),
                patch("src.agents.tool_factory.ColbertRerank") as mock_colbert_class,
            ):
                mock_reranker = MagicMock()
                mock_colbert_class.return_value = mock_reranker

                ToolFactory.create_hybrid_search_tool(mock_retriever)

                # Verify reranker integration
                mock_engine_class.assert_called_once_with(
                    retriever=mock_retriever, node_postprocessors=[mock_reranker]
                )

    def test_create_hybrid_search_tool_none_retriever(self):
        """Test hybrid search tool creation with None retriever."""
        result = ToolFactory.create_hybrid_search_tool(None)

        # Should still create tool with None retriever
        assert isinstance(result, QueryEngineTool)

    def test_create_hybrid_search_tool_description_content(self):
        """Test that hybrid search tool has comprehensive description."""
        mock_retriever = MagicMock()

        with patch("agents.tool_factory.RetrieverQueryEngine"):
            result = ToolFactory.create_hybrid_search_tool(mock_retriever)

            description = result.metadata.description.lower()
            # Verify key concepts are mentioned in description
            assert "hybrid" in description
            assert "rrf" in description or "reciprocal rank fusion" in description
            assert "dense" in description
            assert "sparse" in description
            assert "colbert" in description
            assert "reranking" in description


class TestToolFactoryHybridVector:
    """Test hybrid vector search tool creation."""

    def test_create_hybrid_vector_tool_success(self):
        """Test successful hybrid vector tool creation."""
        mock_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine

        test_settings = DocMindSettings(top_k=7, debug_mode=True, reranker_model=None)

        with patch("agents.tool_factory.settings", test_settings):
            result = ToolFactory.create_hybrid_vector_tool(mock_index)

            assert isinstance(result, QueryEngineTool)
            assert result.query_engine == mock_query_engine
            assert result.metadata.name == "hybrid_vector_search"
            assert "hybrid" in result.metadata.description.lower()
            assert "dense" in result.metadata.description.lower()
            assert "sparse" in result.metadata.description.lower()

            # Verify configuration
            mock_index.as_query_engine.assert_called_once_with(
                similarity_top_k=7, node_postprocessors=[], verbose=True
            )

    def test_create_hybrid_vector_tool_with_reranker(self):
        """Test hybrid vector tool creation with reranker."""
        mock_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine

        test_settings = DocMindSettings(
            top_k=5,
            debug_mode=False,
            reranker_model="colbert-ir/colbertv2.0",
            reranking_top_k=3,
        )

        with (
            patch("src.agents.tool_factory.app_settings", test_settings),
            patch("src.agents.tool_factory.ColbertRerank") as mock_colbert_class,
        ):
            mock_reranker = MagicMock()
            mock_colbert_class.return_value = mock_reranker

            ToolFactory.create_hybrid_vector_tool(mock_index)

            # Verify reranker integration
            mock_index.as_query_engine.assert_called_once_with(
                top_k=5,
                node_postprocessors=[mock_reranker],
                verbose=False,
            )

    def test_create_hybrid_vector_tool_description_completeness(self):
        """Test that hybrid vector tool description mentions key features."""
        mock_index = MagicMock()
        mock_index.as_query_engine.return_value = MagicMock()

        with patch("agents.tool_factory.settings", DocMindSettings()):
            result = ToolFactory.create_hybrid_vector_tool(mock_index)

            description = result.metadata.description.lower()
            assert "bge-large" in description
            assert "splade++" in description
            assert "colbert" in description
            assert "semantic" in description
            assert "keyword" in description


class TestToolFactoryFromIndexes:
    """Test comprehensive tool creation from indexes."""

    def test_create_tools_from_indexes_all_components(self):
        """Test tool creation with all available components."""
        mock_vector_index = MagicMock()
        mock_kg_index = MagicMock()
        mock_retriever = MagicMock()

        with (
            patch.object(ToolFactory, "create_hybrid_search_tool") as mock_hybrid,
            patch.object(ToolFactory, "create_kg_search_tool") as mock_kg,
            patch.object(ToolFactory, "create_vector_search_tool") as mock_vector,
        ):
            # Configure mock returns
            mock_hybrid_tool = MagicMock()
            mock_kg_tool = MagicMock()
            mock_vector_tool = MagicMock()

            mock_hybrid.return_value = mock_hybrid_tool
            mock_kg.return_value = mock_kg_tool
            mock_vector.return_value = mock_vector_tool

            with patch("agents.tool_factory.logging") as mock_logging:
                result = ToolFactory.create_tools_from_indexes(
                    vector_index=mock_vector_index,
                    kg_index=mock_kg_index,
                    retriever=mock_retriever,
                )

                # Should return all 3 tools: hybrid, KG, and vector
                assert len(result) == 3
                assert mock_hybrid_tool in result
                assert mock_kg_tool in result
                assert mock_vector_tool in result

                # Verify method calls
                mock_hybrid.assert_called_once_with(mock_retriever)
                mock_kg.assert_called_once_with(mock_kg_index)
                mock_vector.assert_called_once_with(mock_vector_index)

                # Verify logging
                mock_logging.info.assert_has_calls(
                    [
                        call("Added hybrid fusion search tool"),
                        call("Added knowledge graph search tool"),
                        call("Added vector search tool"),
                        call("Created 3 tools for agent"),
                    ]
                )

    def test_create_tools_from_indexes_vector_only(self):
        """Test tool creation with only vector index."""
        mock_vector_index = MagicMock()

        with (
            patch.object(
                ToolFactory, "create_hybrid_vector_tool"
            ) as mock_hybrid_vector,
            patch.object(ToolFactory, "create_vector_search_tool") as mock_vector,
            patch("agents.tool_factory.logging") as mock_logging,
        ):
            mock_hybrid_vector_tool = MagicMock()
            mock_vector_tool = MagicMock()

            mock_hybrid_vector.return_value = mock_hybrid_vector_tool
            mock_vector.return_value = mock_vector_tool

            result = ToolFactory.create_tools_from_indexes(
                vector_index=mock_vector_index, kg_index=None, retriever=None
            )

            # Should return 2 tools: hybrid vector (fallback) and basic vector
            assert len(result) == 2
            assert mock_hybrid_vector_tool in result
            assert mock_vector_tool in result

            # Verify fallback to hybrid vector was used
            mock_hybrid_vector.assert_called_once_with(mock_vector_index)
            mock_vector.assert_called_once_with(mock_vector_index)

            # Verify logging shows fallback
            mock_logging.info.assert_has_calls(
                [
                    call("Added hybrid vector search tool (fallback)"),
                    call("Knowledge graph index not available"),
                    call("Added vector search tool"),
                    call("Created 2 tools for agent"),
                ]
            )

    def test_create_tools_from_indexes_no_vector_index(self):
        """Test tool creation with no vector index (error case)."""
        with patch("agents.tool_factory.logging") as mock_logging:
            result = ToolFactory.create_tools_from_indexes(
                vector_index=None, kg_index=MagicMock(), retriever=MagicMock()
            )

            # Should return empty list and log error
            assert result == []
            mock_logging.error.assert_called_once_with(
                "Vector index is required for tool creation"
            )

    def test_create_tools_from_indexes_kg_returns_none(self):
        """Test tool creation when KG tool creation returns None."""
        mock_vector_index = MagicMock()
        mock_kg_index = MagicMock()

        with (
            patch.object(
                ToolFactory, "create_hybrid_vector_tool"
            ) as mock_hybrid_vector,
            patch.object(ToolFactory, "create_kg_search_tool", return_value=None),
            patch.object(ToolFactory, "create_vector_search_tool") as mock_vector,
        ):
            mock_hybrid_vector_tool = MagicMock()
            mock_vector_tool = MagicMock()

            mock_hybrid_vector.return_value = mock_hybrid_vector_tool
            mock_vector.return_value = mock_vector_tool

            result = ToolFactory.create_tools_from_indexes(
                vector_index=mock_vector_index,
                kg_index=mock_kg_index,
                retriever=None,
            )

            # Should return 2 tools (no KG tool since it returned None)
            assert len(result) == 2

    def test_create_tools_from_indexes_tool_priority_order(self):
        """Test that tools are created in correct priority order."""
        mock_vector_index = MagicMock()
        mock_kg_index = MagicMock()
        mock_retriever = MagicMock()

        # Create distinct mock tools
        mock_hybrid_tool = MagicMock()
        mock_hybrid_tool.metadata.name = "hybrid_fusion_search"

        mock_kg_tool = MagicMock()
        mock_kg_tool.metadata.name = "knowledge_graph"

        mock_vector_tool = MagicMock()
        mock_vector_tool.metadata.name = "vector_search"

        with (
            patch.object(
                ToolFactory, "create_hybrid_search_tool", return_value=mock_hybrid_tool
            ),
            patch.object(
                ToolFactory, "create_kg_search_tool", return_value=mock_kg_tool
            ),
            patch.object(
                ToolFactory, "create_vector_search_tool", return_value=mock_vector_tool
            ),
        ):
            result = ToolFactory.create_tools_from_indexes(
                vector_index=mock_vector_index,
                kg_index=mock_kg_index,
                retriever=mock_retriever,
            )

            # Verify order: hybrid first, then KG, then vector
            assert len(result) == 3
            assert result[0] == mock_hybrid_tool
            assert result[1] == mock_kg_tool
            assert result[2] == mock_vector_tool


class TestToolFactoryBasicTools:
    """Test basic tools creation (legacy compatibility)."""

    def test_create_basic_tools_success(self):
        """Test successful basic tools creation from index data dict."""
        index_data = {
            "vector": MagicMock(),
            "kg": MagicMock(),
            "retriever": MagicMock(),
        }

        with patch.object(ToolFactory, "create_tools_from_indexes") as mock_create:
            mock_tools = [MagicMock(), MagicMock()]
            mock_create.return_value = mock_tools

            result = ToolFactory.create_basic_tools(index_data)

            assert result == mock_tools
            mock_create.assert_called_once_with(
                vector_index=index_data["vector"],
                kg_index=index_data["kg"],
                retriever=index_data["retriever"],
            )

    def test_create_basic_tools_empty_dict(self):
        """Test basic tools creation with empty index data."""
        with patch.object(ToolFactory, "create_tools_from_indexes") as mock_create:
            mock_create.return_value = []

            ToolFactory.create_basic_tools({})

            # Should pass None for all components
            mock_create.assert_called_once_with(
                vector_index=None, kg_index=None, retriever=None
            )

    def test_create_basic_tools_partial_data(self):
        """Test basic tools creation with partial index data."""
        index_data = {
            "vector": MagicMock(),
            # Missing kg and retriever
        }

        with patch.object(ToolFactory, "create_tools_from_indexes") as mock_create:
            ToolFactory.create_basic_tools(index_data)

            # Should handle missing keys gracefully
            mock_create.assert_called_once_with(
                vector_index=index_data["vector"],
                kg_index=None,  # Default for missing key
                retriever=None,  # Default for missing key
            )

    def test_create_basic_tools_none_values(self):
        """Test basic tools creation with None values in dict."""
        index_data = {"vector": None, "kg": None, "retriever": None}

        with patch.object(ToolFactory, "create_tools_from_indexes") as mock_create:
            ToolFactory.create_basic_tools(index_data)

            mock_create.assert_called_once_with(
                vector_index=None, kg_index=None, retriever=None
            )


class TestToolFactoryEdgeCases:
    """Test edge cases and error scenarios."""

    def test_all_methods_with_mock_settings(self):
        """Test all methods work with various settings configurations."""
        test_cases = [
            # Default settings
            DocMindSettings(),
            # Minimal settings
            DocMindSettings(top_k=1, debug_mode=False, reranker_model=None),
            # Maximal settings
            DocMindSettings(
                top_k=50,
                debug_mode=True,
                reranker_model="colbert-ir/colbertv2.0",
                reranking_top_k=20,
            ),
        ]

        mock_index = MagicMock()
        mock_index.as_query_engine.return_value = MagicMock()

        for settings in test_cases:
            with patch("agents.tool_factory.settings", settings):
                # Test all tool creation methods don't crash
                vector_result = ToolFactory.create_vector_search_tool(mock_index)
                assert isinstance(vector_result, QueryEngineTool)

                kg_result = ToolFactory.create_kg_search_tool(mock_index)
                assert isinstance(kg_result, QueryEngineTool)

                hybrid_vector_result = ToolFactory.create_hybrid_vector_tool(mock_index)
                assert isinstance(hybrid_vector_result, QueryEngineTool)

    def test_tool_metadata_consistency(self):
        """Test that all created tools have consistent metadata structure."""
        mock_index = MagicMock()
        mock_index.as_query_engine.return_value = MagicMock()
        mock_retriever = MagicMock()

        with patch(
            "agents.tool_factory.RetrieverQueryEngine", return_value=MagicMock()
        ):
            tools = [
                ToolFactory.create_vector_search_tool(mock_index),
                ToolFactory.create_kg_search_tool(mock_index),
                ToolFactory.create_hybrid_search_tool(mock_retriever),
                ToolFactory.create_hybrid_vector_tool(mock_index),
            ]

            for tool in tools:
                # All tools should have consistent metadata structure
                assert hasattr(tool.metadata, "name")
                assert hasattr(tool.metadata, "description")
                assert hasattr(tool.metadata, "return_direct")
                assert isinstance(tool.metadata.name, str)
                assert isinstance(tool.metadata.description, str)
                assert isinstance(tool.metadata.return_direct, bool)
                assert tool.metadata.return_direct is False  # All should be False

    def test_query_engine_configuration_isolation(self):
        """Test that query engine configurations don't interfere with each other."""
        mock_index1 = MagicMock()
        mock_index2 = MagicMock()
        mock_query_engine1 = MagicMock()
        mock_query_engine2 = MagicMock()

        mock_index1.as_query_engine.return_value = mock_query_engine1
        mock_index2.as_query_engine.return_value = mock_query_engine2

        test_settings = DocMindSettings(top_k=5, debug_mode=True, reranker_model=None)

        with patch("agents.tool_factory.settings", test_settings):
            # Create two tools from different indexes
            tool1 = ToolFactory.create_vector_search_tool(mock_index1)
            tool2 = ToolFactory.create_kg_search_tool(mock_index2)

            # Each should have its own query engine
            assert tool1.query_engine == mock_query_engine1
            assert tool2.query_engine == mock_query_engine2
            assert tool1.query_engine != tool2.query_engine

            # Verify different configurations were used
            mock_index1.as_query_engine.assert_called_once_with(
                top_k=5, node_postprocessors=[], verbose=True
            )

            mock_index2.as_query_engine.assert_called_once_with(
                top_k=10,  # KG uses higher top_k
                include_text=True,
                node_postprocessors=[],
                verbose=True,
            )


class TestToolFactoryPerformanceScenarios:
    """Test performance-related scenarios."""

    def test_create_tools_from_indexes_performance_logging(self):
        """Test that tool creation includes performance logging."""
        mock_vector_index = MagicMock()

        with (
            patch.object(
                ToolFactory, "create_hybrid_vector_tool"
            ) as mock_hybrid_vector,
            patch.object(ToolFactory, "create_vector_search_tool") as mock_vector,
            patch("agents.tool_factory.logging") as mock_logging,
        ):
            mock_hybrid_vector.return_value = MagicMock()
            mock_vector.return_value = MagicMock()

            ToolFactory.create_tools_from_indexes(vector_index=mock_vector_index)

            # Should log tool creation counts
            mock_logging.info.assert_called_with("Created 2 tools for agent")

    def test_reranker_creation_caching_behavior(self):
        """Test that reranker creation behaves consistently across calls."""
        test_settings = DocMindSettings(
            reranker_model="colbert-ir/colbertv2.0", reranking_top_k=5
        )

        with (
            patch("src.agents.tool_factory.app_settings", test_settings),
            patch("src.agents.tool_factory.ColbertRerank") as mock_colbert_class,
        ):
            mock_reranker1 = MagicMock()
            mock_reranker2 = MagicMock()
            mock_colbert_class.side_effect = [mock_reranker1, mock_reranker2]

            # Create reranker multiple times
            reranker1 = ToolFactory._create_reranker()
            reranker2 = ToolFactory._create_reranker()

            # Each call should create a new instance (no caching)
            assert reranker1 == mock_reranker1
            assert reranker2 == mock_reranker2
            assert reranker1 != reranker2
            assert mock_colbert_class.call_count == 2

    def test_large_tool_creation_batch(self):
        """Test creating many tools doesn't cause issues."""
        mock_vector_index = MagicMock()
        mock_vector_index.as_query_engine.return_value = MagicMock()
        mock_kg_index = MagicMock()
        mock_kg_index.as_query_engine.return_value = MagicMock()
        mock_retriever = MagicMock()

        with (
            patch("agents.tool_factory.RetrieverQueryEngine", return_value=MagicMock()),
            patch("agents.tool_factory.logging"),
        ):
            # Create many tool sets
            for _i in range(10):
                tools = ToolFactory.create_tools_from_indexes(
                    vector_index=mock_vector_index,
                    kg_index=mock_kg_index,
                    retriever=mock_retriever,
                )

                # Each batch should consistently create 3 tools
                assert len(tools) == 3

                # Verify all are QueryEngineTool instances
                for tool in tools:
                    assert isinstance(tool, QueryEngineTool)


class TestToolFactoryIntegration:
    """Test integration scenarios with real-world use cases."""

    def test_complete_agent_tool_setup_scenario(self):
        """Test complete tool setup scenario for agent creation."""
        # Simulate realistic index data
        mock_vector_index = MagicMock()
        mock_vector_index.as_query_engine.return_value = MagicMock()

        mock_kg_index = MagicMock()
        mock_kg_index.as_query_engine.return_value = MagicMock()

        mock_retriever = MagicMock()

        # Realistic settings
        test_settings = DocMindSettings(
            top_k=10,
            debug_mode=False,
            reranker_model="colbert-ir/colbertv2.0",
            reranking_top_k=5,
        )

        with (
            patch("src.agents.tool_factory.app_settings", test_settings),
            patch("src.agents.tool_factory.ColbertRerank") as mock_colbert_class,
            patch("agents.tool_factory.RetrieverQueryEngine", return_value=MagicMock()),
        ):
            mock_reranker = MagicMock()
            mock_colbert_class.return_value = mock_reranker

            # Create complete tool set
            tools = ToolFactory.create_tools_from_indexes(
                vector_index=mock_vector_index,
                kg_index=mock_kg_index,
                retriever=mock_retriever,
            )

            # Verify complete tool set
            assert len(tools) == 3

            # Verify tool names for agent identification
            tool_names = [tool.metadata.name for tool in tools]
            expected_names = [
                "hybrid_fusion_search",
                "knowledge_graph",
                "vector_search",
            ]
            for name in expected_names:
                assert name in tool_names

            # Verify all tools have comprehensive descriptions
            for tool in tools:
                assert len(tool.metadata.description) > 50  # Substantial description
                assert "best for:" in tool.metadata.description.lower()

    def test_tool_factory_with_different_llm_backends(self):
        """Test tool factory works with different LLM backend configurations."""
        mock_index = MagicMock()
        mock_index.as_query_engine.return_value = MagicMock()

        # Test with different configuration scenarios
        configurations = [
            {"debug_mode": True, "reranker_model": None},  # Development
            {
                "debug_mode": False,
                "reranker_model": "colbert-ir/colbertv2.0",
            },  # Production
            {
                "debug_mode": True,
                "reranker_model": "colbert-ir/colbertv2.0",
            },  # Debug with reranker
        ]

        for config in configurations:
            test_settings = DocMindSettings(**config)

            with patch("agents.tool_factory.settings", test_settings):
                if config.get("reranker_model"):
                    with patch(
                        "agents.tool_factory.ColbertRerank", return_value=MagicMock()
                    ):
                        result = ToolFactory.create_vector_search_tool(mock_index)
                else:
                    result = ToolFactory.create_vector_search_tool(mock_index)

                # Should work with all configurations
                assert isinstance(result, QueryEngineTool)
                assert result.metadata.name == "vector_search"
