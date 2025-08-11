"""Comprehensive tests for agent utilities functionality.

This module provides in-depth test coverage for agent utilities, focusing on:
- Tool creation from index data
- ReActAgent creation and configuration
- Agentic document analysis
- Async chat capabilities
- Error handling and resilience
- Performance monitoring

Follows 2025 Python testing best practices with comprehensive scenarios.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama

from agents.agent_utils import (
    create_tools_from_index,
)
from models import AppSettings
from utils.exceptions import ConfigurationError


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return AppSettings(
        reranker_model="jinaai/jina-reranker-v2-base-multilingual",
        reranking_top_k=5,
        rrf_fusion_alpha=60,
        default_model="google/gemma-3n-E4B-it",
    )


@pytest.fixture
def mock_vector_index():
    """Create a mock vector index for testing."""
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_index.as_query_engine.return_value = mock_query_engine
    return mock_index


@pytest.fixture
def mock_kg_index():
    """Create a mock knowledge graph index for testing."""
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_index.as_query_engine.return_value = mock_query_engine
    return mock_index


@pytest.fixture
def mock_hybrid_retriever():
    """Create a mock hybrid retriever for testing."""
    return MagicMock()


@pytest.fixture
def mock_index_data_complete(mock_vector_index, mock_kg_index, mock_hybrid_retriever):
    """Create complete index data with all components."""
    return {
        "vector": mock_vector_index,
        "kg": mock_kg_index,
        "retriever": mock_hybrid_retriever,
    }


@pytest.fixture
def mock_index_data_vector_only(mock_vector_index):
    """Create index data with vector index only."""
    return {
        "vector": mock_vector_index,
        "kg": None,
        "retriever": None,
    }


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock_llm = MagicMock(spec=Ollama)
    mock_llm.model = "google/gemma-3n-E4B-it"
    return mock_llm


@pytest.fixture
def mock_agent():
    """Create a mock ReActAgent for testing."""
    mock_agent = MagicMock(spec=ReActAgent)
    mock_response = MagicMock()
    mock_response.response = "Mock agent response"
    mock_agent.chat.return_value = mock_response

    # Mock async stream chat
    mock_async_response = MagicMock()
    mock_async_response.async_response_gen = AsyncMock(
        return_value=["Mock ", "streaming ", "response"]
    )
    mock_agent.async_stream_chat = AsyncMock(return_value=mock_async_response)

    return mock_agent


class TestCreateToolsFromIndex:
    """Test tool creation from indexes with hybrid search and reranking."""

    @pytest.mark.parametrize(
        ("test_data", "expected_count", "reranker_model", "should_have_kg"),
        [
            (
                {
                    "vector": MagicMock(),
                    "kg": MagicMock(),
                    "retriever": MagicMock(),
                },
                2,
                "test_reranker_model",
                True,
            ),
            (
                {"vector": MagicMock()},
                1,
                None,
                False,
            ),
        ],
    )
    @patch("agents.agent_utils.ColbertRerank")
    @patch("agents.agent_utils.RetrieverQueryEngine")
    @patch("agents.agent_utils.QueryEngineTool")
    def test_create_tools_with_hybrid_configuration(
        self,
        mock_query_tool,
        mock_retriever_query_engine,
        mock_colbert_rerank,
        mock_settings,
        test_data,
        expected_count,
        reranker_model,
        should_have_kg,
    ):
        """Test tool creation with hybrid configuration and variations."""
        # Override reranker model
        mock_settings.reranker_model = reranker_model

        # Mock reranker
        mock_reranker = MagicMock()
        mock_colbert_rerank.return_value = mock_reranker

        # Mock hybrid query engine
        mock_hybrid_engine = MagicMock()
        mock_retriever_query_engine.return_value = mock_hybrid_engine

        # Mock query engine tools
        mock_tools = []
        for _ in range(expected_count):
            tool = MagicMock()
            tool.metadata.name = (
                "hybrid_fusion_search"
                if len(mock_tools) == 0
                else "knowledge_graph_query"
            )
            mock_tools.append(tool)
        mock_query_tool.side_effect = mock_tools

        with patch("agents.agent_utils.settings", mock_settings):
            # Actual function call
            tools = create_tools_from_index(test_data)

            # Common verifications
            assert len(tools) == expected_count
            assert mock_query_tool.call_count == expected_count

            # Verify reranker configuration
            if reranker_model:
                mock_colbert_rerank.assert_called_once_with(
                    model=reranker_model,
                    top_n=mock_settings.reranking_top_k,
                    keep_retrieval_score=True,
                )

            # Verify tool metadata and configuration for complete configuration
            if should_have_kg:
                # Verify hybrid query engine
                mock_retriever_query_engine.assert_called_once_with(
                    retriever=test_data["retriever"],
                    node_postprocessors=[mock_reranker],
                )

                # Verify KG query engine
                test_data["kg"].as_query_engine.assert_called_once_with(
                    similarity_top_k=10,
                    include_text=True,
                    node_postprocessors=[mock_reranker],
                )

    def test_create_tools_empty_index_data(self, mock_settings):
        """Test tool creation with empty index data."""
        with (
            patch("agents.agent_utils.settings", mock_settings),
            pytest.raises(ConfigurationError) as exc_info,
        ):
            create_tools_from_index({})

        assert exc_info.value.context["index_data_keys"] == []
        assert exc_info.value.operation == "tool_creation_validation"

    def test_create_tools_none_data(self, mock_settings):
        """Test tool creation with None index data."""
        with (
            patch("agents.agent_utils.settings", mock_settings),
            pytest.raises(ConfigurationError),
        ):
            create_tools_from_index(None)
