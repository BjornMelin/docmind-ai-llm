"""Enhanced unit tests for tool_factory.py with realistic scenarios.

Focuses on testing actual ToolFactory behavior with minimal mocking,
realistic test data, and comprehensive edge case coverage.
Tests tool creation logic rather than mock interactions.
"""

import os
from unittest.mock import Mock, patch

import pytest
from llama_index.core.tools import QueryEngineTool

from src.agents.tool_factory import (
    DEFAULT_RERANKING_TOP_K,
    DEFAULT_VECTOR_SIMILARITY_TOP_K,
    KG_SIMILARITY_TOP_K,
    ToolFactory,
)
from src.config.settings import DocMindSettings

# Performance tests require explicit opt-in
pytestmark = pytest.mark.skipif(os.getenv("PYTEST_PERF") != "1", reason="perf disabled")


@pytest.fixture
def mock_index():
    """Create a lightweight mock index for testing."""
    index = Mock()
    query_engine = Mock()
    index.as_query_engine.return_value = query_engine
    return index


@pytest.fixture
def mock_retriever():
    """Create a lightweight mock retriever for testing."""
    return Mock()


@pytest.fixture
def basic_settings() -> DocMindSettings:
    """Create basic test settings."""
    return DocMindSettings()


@pytest.fixture
def reranker_settings() -> DocMindSettings:
    """Create settings with reranker enabled."""
    return DocMindSettings(
        reranker_model="colbert-ir/colbertv2.0",
        reranking_top_k=5,
    )


class TestToolFactoryConstants:
    """Test ToolFactory constants are reasonable."""

    def test_constants_are_valid(self):
        """Test that ToolFactory constants have reasonable values."""
        assert isinstance(DEFAULT_RERANKING_TOP_K, int)
        assert DEFAULT_RERANKING_TOP_K > 0

        assert isinstance(KG_SIMILARITY_TOP_K, int)
        assert KG_SIMILARITY_TOP_K > 0

        assert isinstance(DEFAULT_VECTOR_SIMILARITY_TOP_K, int)
        assert DEFAULT_VECTOR_SIMILARITY_TOP_K > 0

        # KG should typically need more results than vector search
        assert KG_SIMILARITY_TOP_K >= DEFAULT_VECTOR_SIMILARITY_TOP_K


class TestToolFactoryBasicMethods:
    """Test basic ToolFactory methods with realistic scenarios."""

    def test_create_query_tool_basic(self):
        """Test basic query tool creation works correctly."""
        mock_engine = Mock()
        tool_name = "test_tool"
        tool_description = "A test tool for validation"

        result = ToolFactory.create_query_tool(mock_engine, tool_name, tool_description)

        assert isinstance(result, QueryEngineTool)
        assert result.query_engine == mock_engine
        assert result.metadata.name == tool_name
        assert result.metadata.description == tool_description
        assert result.metadata.return_direct is False  # Should allow agent reasoning

    def test_create_query_tool_edge_cases(self):
        """Test query tool creation with edge case inputs."""
        # Test with None engine
        result = ToolFactory.create_query_tool(None, "test", "description")
        assert isinstance(result, QueryEngineTool)
        assert result.query_engine is None

        # Test with empty strings
        result = ToolFactory.create_query_tool(Mock(), "", "")
        assert result.metadata.name == ""
        assert result.metadata.description == ""

        # Test with very long inputs
        long_name = "x" * 1000
        long_desc = "y" * 10000
        result = ToolFactory.create_query_tool(Mock(), long_name, long_desc)
        assert result.metadata.name == long_name
        assert result.metadata.description == long_desc

    @pytest.mark.parametrize(
        ("name", "description"),
        [
            ("simple_tool", "Simple description"),
            ("tool-with-dashes", "Tool with dashes"),
            ("tool_with_underscores", "Tool with underscores"),
            ("123numeric", "Numeric start"),
            ("UPPERCASE", "Uppercase name"),
        ],
    )
    def test_create_query_tool_name_variations(self, name, description):
        """Test query tool creation with various name patterns."""
        mock_engine = Mock()

        result = ToolFactory.create_query_tool(mock_engine, name, description)

        assert result.metadata.name == name
        assert result.metadata.description == description

    def test_create_query_tool_metadata_consistency(self):
        """Test that query tool metadata is consistent."""
        mock_engine = Mock()

        tool = ToolFactory.create_query_tool(
            mock_engine, "test_tool", "Test description"
        )

        # Verify metadata structure
        assert hasattr(tool.metadata, "name")
        assert hasattr(tool.metadata, "description")
        assert hasattr(tool.metadata, "return_direct")

        # Verify types
        assert isinstance(tool.metadata.name, str)
        assert isinstance(tool.metadata.description, str)
        assert isinstance(tool.metadata.return_direct, bool)


class TestToolFactoryReranker:
    """Test ColBERT reranker creation with realistic scenarios."""

    def test_create_reranker_with_valid_settings(self):
        """Test reranker creation with valid configuration."""
        test_settings = DocMindSettings(
            reranker_model="colbert-ir/colbertv2.0",
            reranking_top_k=5,
        )

        with (
            patch("src.agents.tool_factory.settings", test_settings),
            patch("src.agents.tool_factory.ColbertRerank") as mock_colbert,
        ):
            mock_reranker = Mock()
            mock_colbert.return_value = mock_reranker

            result = ToolFactory._create_reranker()

            assert result == mock_reranker
            mock_colbert.assert_called_once_with(
                top_n=5, model="colbert-ir/colbertv2.0", keep_retrieval_score=True
            )

    def test_create_reranker_no_model_configured(self):
        """Test reranker creation when no model is configured."""
        test_settings = DocMindSettings(reranker_model=None)

        with patch("src.agents.tool_factory.settings", test_settings):
            result = ToolFactory._create_reranker()
            assert result is None

    def test_create_reranker_empty_model_string(self):
        """Test reranker creation with empty model string."""
        test_settings = DocMindSettings(reranker_model="")

        with patch("src.agents.tool_factory.settings", test_settings):
            result = ToolFactory._create_reranker()
            assert result is None

    def test_create_reranker_uses_default_top_k(self):
        """Test reranker creation uses default top_k when not specified."""
        test_settings = DocMindSettings(
            reranker_model="colbert-ir/colbertv2.0",
            reranking_top_k=None,  # Should use default
        )

        with (
            patch("src.agents.tool_factory.settings", test_settings),
            patch("src.agents.tool_factory.ColbertRerank") as mock_colbert,
        ):
            mock_reranker = Mock()
            mock_colbert.return_value = mock_reranker

            ToolFactory._create_reranker()

            # Should use DEFAULT_RERANKING_TOP_K
            mock_colbert.assert_called_once_with(
                top_n=DEFAULT_RERANKING_TOP_K,
                model="colbert-ir/colbertv2.0",
                keep_retrieval_score=True,
            )

    def test_create_reranker_error_handling(self):
        """Test reranker creation handles errors gracefully."""
        test_settings = DocMindSettings(
            reranker_model="colbert-ir/colbertv2.0",
            reranking_top_k=5,
        )

        # Test import error
        with (
            patch("src.agents.tool_factory.settings", test_settings),
            patch(
                "src.agents.tool_factory.ColbertRerank",
                side_effect=ImportError("ColbertRerank not available"),
            ),
        ):
            result = ToolFactory._create_reranker()
            assert result is None

        # Test runtime error
        with (
            patch("src.agents.tool_factory.settings", test_settings),
            patch(
                "src.agents.tool_factory.ColbertRerank",
                side_effect=RuntimeError("Model loading failed"),
            ),
        ):
            result = ToolFactory._create_reranker()
            assert result is None

    def test_reranker_parameter_validation(self):
        """Test reranker parameter handling."""
        # Test various top_k values
        for top_k in [1, 5, 10, 20]:
            test_settings = DocMindSettings(
                reranker_model="colbert-ir/colbertv2.0",
                reranking_top_k=top_k,
            )

            with (
                patch("src.agents.tool_factory.settings", test_settings),
                patch("src.agents.tool_factory.ColbertRerank") as mock_colbert,
            ):
                mock_colbert.return_value = Mock()
                ToolFactory._create_reranker()
                mock_colbert.assert_called_once_with(
                    top_n=top_k,
                    model="colbert-ir/colbertv2.0",
                    keep_retrieval_score=True,
                )


class TestToolFactoryVectorSearch:
    """Test vector search tool creation with realistic scenarios."""

    def test_create_vector_search_tool_basic(self, mock_index, basic_settings):
        """Test basic vector search tool creation."""
        with patch("src.agents.tool_factory.settings", basic_settings):
            tool = ToolFactory.create_vector_search_tool(mock_index)

            assert isinstance(tool, QueryEngineTool)
            assert tool.metadata.name == "vector_search"
            assert "semantic similarity" in tool.metadata.description.lower()
            assert tool.query_engine is not None

            # Verify index configuration
            mock_index.as_query_engine.assert_called_once()

    def test_create_vector_search_tool_with_custom_settings(self, mock_index):
        """Test vector search tool with custom similarity_top_k."""
        custom_settings = DocMindSettings(top_k=10, debug=True)

        with patch("src.agents.tool_factory.settings", custom_settings):
            ToolFactory.create_vector_search_tool(mock_index)

            # Should use custom settings
            call_args = mock_index.as_query_engine.call_args
            assert call_args[1]["similarity_top_k"] == 10
            assert call_args[1]["verbose"] is True


class TestToolFactoryKnowledgeGraph:
    """Test knowledge graph tool creation."""

    def test_create_kg_search_tool_success(self, basic_settings):
        """Test successful KG search tool creation."""
        mock_kg_index = Mock()
        mock_query_engine = Mock()
        mock_kg_index.as_query_engine.return_value = mock_query_engine

        with patch("src.agents.tool_factory.settings", basic_settings):
            tool = ToolFactory.create_kg_search_tool(mock_kg_index)

            assert isinstance(tool, QueryEngineTool)
            assert tool.metadata.name == "knowledge_graph"
            assert "knowledge graph" in tool.metadata.description.lower()
            assert "entity" in tool.metadata.description.lower()

            # Verify KG-specific configuration
            call_args = mock_kg_index.as_query_engine.call_args
            assert call_args[1]["similarity_top_k"] == KG_SIMILARITY_TOP_K
            assert call_args[1]["include_text"] is True

    def test_create_kg_search_tool_none_index(self):
        """Test KG search tool creation with None index."""
        result = ToolFactory.create_kg_search_tool(None)
        assert result is None

    def test_create_kg_search_tool_falsy_values(self):
        """Test KG search tool with various falsy values."""
        falsy_values = [None, False, 0, "", [], {}]

        for falsy_value in falsy_values:
            result = ToolFactory.create_kg_search_tool(falsy_value)
            assert result is None


class TestToolFactoryHybridSearch:
    """Test hybrid search tool creation."""

    def test_create_hybrid_search_tool_success(self, mock_retriever, basic_settings):
        """Test successful hybrid search tool creation."""
        with (
            patch("src.agents.tool_factory.settings", basic_settings),
            patch("src.agents.tool_factory.RetrieverQueryEngine") as mock_engine_class,
        ):
            mock_query_engine = Mock()
            mock_engine_class.return_value = mock_query_engine

            tool = ToolFactory.create_hybrid_search_tool(mock_retriever)

            assert isinstance(tool, QueryEngineTool)
            assert tool.metadata.name == "hybrid_fusion_search"
            assert "rrf" in tool.metadata.description.lower()
            assert "reciprocal rank fusion" in tool.metadata.description.lower()

            # Verify engine configuration
            mock_engine_class.assert_called_once_with(
                retriever=mock_retriever, node_postprocessors=[]
            )

    def test_create_hybrid_search_tool_with_none_retriever(self):
        """Test hybrid search tool with None retriever."""
        with patch("src.agents.tool_factory.RetrieverQueryEngine") as mock_engine_class:
            mock_engine_class.return_value = Mock()

            tool = ToolFactory.create_hybrid_search_tool(None)

            assert isinstance(tool, QueryEngineTool)
            mock_engine_class.assert_called_once_with(
                retriever=None, node_postprocessors=[]
            )


class TestToolFactoryIntegration:
    """Test tool factory integration scenarios."""

    def test_create_tools_from_indexes_comprehensive(self, mock_index, mock_retriever):
        """Test creating tools from all available components."""
        mock_kg_index = Mock()
        mock_kg_index.as_query_engine.return_value = Mock()

        with patch("src.agents.tool_factory.settings", DocMindSettings()):
            tools = ToolFactory.create_tools_from_indexes(
                vector_index=mock_index,
                kg_index=mock_kg_index,
                retriever=mock_retriever,
            )

            # Should create hybrid fusion, KG, and vector tools
            assert len(tools) >= 2  # At least hybrid + vector

            tool_names = [tool.metadata.name for tool in tools]
            assert "hybrid_fusion_search" in tool_names
            assert "vector_search" in tool_names

    def test_create_tools_from_indexes_vector_only(self, mock_index):
        """Test tool creation with only vector index."""
        with patch("src.agents.tool_factory.settings", DocMindSettings()):
            tools = ToolFactory.create_tools_from_indexes(
                vector_index=mock_index, kg_index=None, retriever=None
            )

            # Should create hybrid vector + basic vector tools
            assert len(tools) == 2
            tool_names = [tool.metadata.name for tool in tools]
            assert "hybrid_vector_search" in tool_names
            assert "vector_search" in tool_names

    def test_create_tools_from_indexes_no_vector_index(self):
        """Test error handling when no vector index provided."""
        tools = ToolFactory.create_tools_from_indexes(
            vector_index=None, kg_index=Mock(), retriever=Mock()
        )

        # Should return empty list
        assert tools == []

    def test_create_basic_tools_legacy_compatibility(self, mock_index):
        """Test legacy create_basic_tools method."""
        index_data = {"vector": mock_index, "kg": None, "retriever": None}

        tools = ToolFactory.create_basic_tools(index_data)

        # Should work the same as create_tools_from_indexes
        assert isinstance(tools, list)
        assert len(tools) >= 1

    def test_create_basic_tools_empty_dict(self):
        """Test basic tools creation with empty dict."""
        tools = ToolFactory.create_basic_tools({})
        assert tools == []  # Should return empty list when no vector index


class TestToolFactoryEdgeCases:
    """Test edge cases and error conditions."""

    def test_tool_descriptions_are_informative(self, mock_index):
        """Test that tool descriptions contain helpful information."""
        with patch("src.agents.tool_factory.settings", DocMindSettings()):
            vector_tool = ToolFactory.create_vector_search_tool(mock_index)
            hybrid_tool = ToolFactory.create_hybrid_vector_tool(mock_index)

            # Descriptions should be substantial and informative
            assert len(vector_tool.metadata.description) > 50
            assert len(hybrid_tool.metadata.description) > 50

            # Should contain key terms
            assert "best for:" in vector_tool.metadata.description.lower()
            assert "semantic" in vector_tool.metadata.description.lower()

    def test_tool_metadata_consistency(self, mock_index, mock_retriever):
        """Test that all tools have consistent metadata structure."""
        with (
            patch("src.agents.tool_factory.settings", DocMindSettings()),
            patch("src.agents.tool_factory.RetrieverQueryEngine", return_value=Mock()),
        ):
            tools = [
                ToolFactory.create_vector_search_tool(mock_index),
                ToolFactory.create_hybrid_vector_tool(mock_index),
                ToolFactory.create_hybrid_search_tool(mock_retriever),
            ]

            for tool in tools:
                assert hasattr(tool.metadata, "name")
                assert hasattr(tool.metadata, "description")
                assert hasattr(tool.metadata, "return_direct")
                assert isinstance(tool.metadata.name, str)
                assert isinstance(tool.metadata.description, str)
                assert tool.metadata.return_direct is False

    @pytest.mark.performance
    def test_tool_creation_performance(self):
        """Test that tool creation is reasonably fast."""
        import time

        mock_engine = Mock()
        start_time = time.time()

        # Create many tools
        for i in range(100):
            ToolFactory.create_query_tool(mock_engine, f"tool_{i}", f"Description {i}")

        elapsed_time = time.time() - start_time

        # Should create 100 tools in under 1 second
        assert elapsed_time < 1.0

    def test_error_resilience(self):
        """Test that tool factory handles errors gracefully."""
        # Should handle None inputs gracefully
        tool = ToolFactory.create_query_tool(None, "test", "test")
        assert isinstance(tool, QueryEngineTool)

        # Should handle empty strings
        tool = ToolFactory.create_query_tool(Mock(), "", "")
        assert isinstance(tool, QueryEngineTool)
