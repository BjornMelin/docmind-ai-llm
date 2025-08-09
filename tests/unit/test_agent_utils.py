"""Comprehensive tests for agent utilities functionality.

This module tests agent creation, tool configuration, hybrid search integration,
knowledge graph functionality, ReActAgent creation, and async chat capabilities
following 2025 best practices.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama

from agents.agent_utils import (
    analyze_documents_agentic,
    chat_with_agent,
    create_agent_with_tools,
    create_tools_from_index,
)
from models import AppSettings


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

    @patch("agents.agent_utils.ColbertRerank")
    @patch("agents.agent_utils.RetrieverQueryEngine")
    @patch("agents.agent_utils.QueryEngineTool")
    def test_create_tools_with_hybrid_retriever_and_kg(
        self,
        mock_query_tool,
        mock_retriever_query_engine,
        mock_colbert_rerank,
        mock_index_data_complete,
        mock_settings,
    ):
        """Test tool creation with hybrid retriever and knowledge graph."""
        # Mock ColBERT reranker
        mock_reranker = MagicMock()
        mock_colbert_rerank.return_value = mock_reranker

        # Mock hybrid query engine
        mock_hybrid_engine = MagicMock()
        mock_retriever_query_engine.return_value = mock_hybrid_engine

        # Mock query engine tools
        mock_hybrid_tool = MagicMock()
        mock_kg_tool = MagicMock()
        mock_query_tool.side_effect = [mock_hybrid_tool, mock_kg_tool]

        with (
            patch("agents.agent_utils.settings", mock_settings),
            patch("agents.agent_utils.logging.info") as mock_log_info,
        ):
            tools = create_tools_from_index(mock_index_data_complete)

            # Verify ColBERT reranker creation
            mock_colbert_rerank.assert_called_once_with(
                model=mock_settings.reranker_model,
                top_n=mock_settings.reranking_top_k,
                keep_retrieval_score=True,
            )

            # Verify hybrid query engine creation
            mock_retriever_query_engine.assert_called_once_with(
                retriever=mock_index_data_complete["retriever"],
                node_postprocessors=[mock_reranker],
            )

            # Verify KG query engine creation
            mock_index_data_complete["kg"].as_query_engine.assert_called_once_with(
                similarity_top_k=10,
                include_text=True,
                node_postprocessors=[mock_reranker],
            )

            # Verify tool creation calls
            assert mock_query_tool.call_count == 2

            # Verify hybrid fusion search tool
            hybrid_call_args = mock_query_tool.call_args_list[0]
            assert hybrid_call_args[1]["query_engine"] == mock_hybrid_engine
            hybrid_metadata = hybrid_call_args[1]["metadata"]
            assert hybrid_metadata.name == "hybrid_fusion_search"
            assert "QueryFusionRetriever" in hybrid_metadata.description
            assert "RRF" in hybrid_metadata.description
            assert "ColBERT" in hybrid_metadata.description

            # Verify KG query tool
            kg_call_args = mock_query_tool.call_args_list[1]
            kg_metadata = kg_call_args[1]["metadata"]
            assert kg_metadata.name == "knowledge_graph_query"
            assert "entity and relationship" in kg_metadata.description

            # Verify logging
            assert len(tools) == 2
            log_messages = [call[0][0] for call in mock_log_info.call_args_list]
            assert any("Hybrid fusion search tool added" in msg for msg in log_messages)
            assert any(
                "Knowledge Graph query tool added" in msg for msg in log_messages
            )

    @patch("agents.agent_utils.ColbertRerank")
    @patch("agents.agent_utils.QueryEngineTool")
    def test_create_tools_vector_only_fallback(
        self,
        mock_query_tool,
        mock_colbert_rerank,
        mock_index_data_vector_only,
        mock_settings,
    ):
        """Test tool creation with vector index only (fallback mode)."""
        # Mock ColBERT reranker
        mock_reranker = MagicMock()
        mock_colbert_rerank.return_value = mock_reranker

        # Mock vector query engine
        mock_vector_engine = MagicMock()
        mock_index_data_vector_only[
            "vector"
        ].as_query_engine.return_value = mock_vector_engine

        # Mock query engine tool
        mock_vector_tool = MagicMock()
        mock_query_tool.return_value = mock_vector_tool

        with (
            patch("agents.agent_utils.settings", mock_settings),
            patch("agents.agent_utils.logging.info") as mock_log_info,
            patch("agents.agent_utils.logging.warning") as mock_log_warning,
        ):
            tools = create_tools_from_index(mock_index_data_vector_only)

            # Verify fallback vector query engine creation
            mock_index_data_vector_only[
                "vector"
            ].as_query_engine.assert_called_once_with(
                similarity_top_k=mock_settings.reranking_top_k,
                hybrid_alpha=mock_settings.rrf_fusion_alpha,
                node_postprocessors=[mock_reranker],
            )

            # Verify single tool creation
            assert mock_query_tool.call_count == 1
            vector_call_args = mock_query_tool.call_args
            assert vector_call_args[1]["query_engine"] == mock_vector_engine
            vector_metadata = vector_call_args[1]["metadata"]
            assert vector_metadata.name == "hybrid_vector_search"
            assert "BGE-Large" in vector_metadata.description
            assert "SPLADE++" in vector_metadata.description

            # Verify logging
            assert len(tools) == 1
            mock_log_info.assert_called_with("Fallback hybrid vector search tool added")
            mock_log_warning.assert_called_with(
                "Knowledge Graph index not available - only vector search will be used"
            )

    @patch("agents.agent_utils.QueryEngineTool")
    def test_create_tools_no_reranker(
        self, mock_query_tool, mock_index_data_vector_only, mock_settings
    ):
        """Test tool creation without reranker configuration."""
        # Set reranker model to None
        mock_settings.reranker_model = None

        mock_vector_tool = MagicMock()
        mock_query_tool.return_value = mock_vector_tool

        with patch("agents.agent_utils.settings", mock_settings):
            tools = create_tools_from_index(mock_index_data_vector_only)

            # Verify vector query engine created without postprocessors
            mock_index_data_vector_only[
                "vector"
            ].as_query_engine.assert_called_once_with(
                similarity_top_k=mock_settings.reranking_top_k,
                hybrid_alpha=mock_settings.rrf_fusion_alpha,
                node_postprocessors=[],  # Empty postprocessors
            )

            assert len(tools) == 1

    def test_create_tools_empty_index_data(self, mock_settings):
        """Test tool creation with empty index data."""
        empty_index_data = {"vector": None, "kg": None, "retriever": None}

        with (
            patch("agents.agent_utils.settings", mock_settings),
            pytest.raises(AttributeError),
        ):
            create_tools_from_index(empty_index_data)


class TestCreateAgentWithTools:
    """Test ReActAgent creation with enhanced tools."""

    @patch("agents.agent_utils.create_tools_from_index")
    @patch("agents.agent_utils.ReActAgent.from_tools")
    @patch("agents.agent_utils.ChatMemoryBuffer.from_defaults")
    def test_create_agent_success(
        self,
        mock_memory_buffer,
        mock_react_agent,
        mock_create_tools,
        mock_index_data_complete,
        mock_llm,
    ):
        """Test successful ReActAgent creation with enhanced tools."""
        # Mock tools creation
        mock_tool1 = MagicMock()
        mock_tool1.metadata.name = "hybrid_fusion_search"
        mock_tool2 = MagicMock()
        mock_tool2.metadata.name = "knowledge_graph_query"
        mock_tools = [mock_tool1, mock_tool2]
        mock_create_tools.return_value = mock_tools

        # Mock memory buffer
        mock_memory = MagicMock()
        mock_memory_buffer.return_value = mock_memory

        # Mock ReActAgent creation
        mock_agent = MagicMock(spec=ReActAgent)
        mock_react_agent.return_value = mock_agent

        with patch("agents.agent_utils.logging.info") as mock_log_info:
            create_agent_with_tools(mock_index_data_complete, mock_llm)

            # Verify tools creation
            mock_create_tools.assert_called_once_with(mock_index_data_complete)

            # Verify memory buffer creation
            mock_memory_buffer.assert_called_once_with(token_limit=8192)

            # Verify ReActAgent creation
            mock_react_agent.assert_called_once_with(
                tools=mock_tools,
                llm=mock_llm,
                verbose=True,
                max_iterations=10,
                memory=mock_memory,
            )

            # Verify logging
            log_messages = [call[0][0] for call in mock_log_info.call_args_list]
            assert any(
                "ReActAgent created with 2 enhanced tools" in msg
                for msg in log_messages
            )
            assert any(
                "Tools available: hybrid_fusion_search, knowledge_graph_query" in msg
                for msg in log_messages
            )

            assert result == mock_agent

    @patch("agents.agent_utils.create_tools_from_index")
    @patch("agents.agent_utils.ReActAgent.from_tools")
    def test_create_agent_fallback_on_error(
        self, mock_react_agent, mock_create_tools, mock_index_data_complete, mock_llm
    ):
        """Test ReActAgent creation with fallback on configuration error."""
        # Mock tools creation
        mock_tool = MagicMock()
        mock_tool.metadata.name = "test_tool"
        mock_tools = [mock_tool]
        mock_create_tools.return_value = mock_tools

        # Mock ReActAgent creation failure then success
        mock_fallback_agent = MagicMock()
        mock_react_agent.side_effect = [
            Exception("Enhanced config failed"),
            mock_fallback_agent,
        ]

        with (
            patch("agents.agent_utils.logging.error") as mock_log_error,
            patch("agents.agent_utils.logging.warning") as mock_log_warning,
        ):
            create_agent_with_tools(mock_index_data_complete, mock_llm)

            # Verify error handling
            mock_log_error.assert_called_once()
            error_message = mock_log_error.call_args[0][0]
            assert "ReActAgent creation failed" in error_message

            # Verify fallback agent creation
            assert mock_react_agent.call_count == 2
            fallback_call = mock_react_agent.call_args_list[1]
            assert fallback_call[0] == (mock_tools, mock_llm)  # Positional args
            assert fallback_call[1]["verbose"] is True

            # Verify fallback warning
            mock_log_warning.assert_called_once()
            warning_message = mock_log_warning.call_args[0][0]
            assert "Using fallback ReActAgent configuration" in warning_message
            assert "test_tool" in warning_message

            assert result == mock_fallback_agent

    @patch("agents.agent_utils.create_tools_from_index")
    @patch("agents.agent_utils.ReActAgent.from_tools")
    def test_create_agent_no_tools(
        self, mock_react_agent, mock_create_tools, mock_index_data_complete, mock_llm
    ):
        """Test ReActAgent creation with no tools available."""
        # Mock empty tools creation
        mock_create_tools.return_value = []

        # Mock ReActAgent creation failure then success
        mock_fallback_agent = MagicMock()
        mock_react_agent.side_effect = [
            Exception("No tools provided"),
            mock_fallback_agent,
        ]

        with patch("agents.agent_utils.logging.warning") as mock_log_warning:
            create_agent_with_tools(mock_index_data_complete, mock_llm)

            # Verify warning about empty tools
            warning_messages = [call[0][0] for call in mock_log_warning.call_args_list]
            fallback_warning = next(
                (msg for msg in warning_messages if "Using fallback ReActAgent" in msg),
                None,
            )
            assert fallback_warning is not None
            assert "with tools: " in fallback_warning  # Empty tools list

            assert result == mock_fallback_agent


class TestAnalyzeDocumentsAgentic:
    """Test agentic document analysis functionality."""

    @patch("agents.agent_utils.QueryEngineTool")
    @patch("agents.agent_utils.ReActAgent.from_tools")
    def test_analyze_documents_with_agent(
        self, mock_react_agent, mock_query_tool, mock_agent, mock_index_data_complete
    ):
        """Test agentic analysis with provided agent."""
        # Mock response
        mock_response = MagicMock()
        mock_response.response = "Comprehensive analysis result"
        mock_agent.chat.return_value = mock_response

        analyze_documents_agentic(
            mock_agent, mock_index_data_complete, "Comprehensive Document Analysis"
        )

        # Verify agent chat was called
        mock_agent.chat.assert_called_once_with(
            "Analyze with prompt: Comprehensive Document Analysis"
        )

        # Verify response extraction
        assert result == "Comprehensive analysis result"

        # Verify ReActAgent.from_tools was not called (agent provided)
        mock_react_agent.assert_not_called()

    @patch("agents.agent_utils.QueryEngineTool")
    @patch("agents.agent_utils.ReActAgent.from_tools")
    @patch("agents.agent_utils.Ollama")
    def test_analyze_documents_without_agent(
        self,
        mock_ollama,
        mock_react_agent,
        mock_query_tool,
        mock_index_data_complete,
        mock_settings,
    ):
        """Test agentic analysis with agent creation fallback."""
        # Mock tools
        mock_vector_tool = MagicMock()
        mock_kg_tool = MagicMock()
        mock_query_tool.side_effect = [mock_vector_tool, mock_kg_tool]

        # Mock Ollama LLM
        mock_llm = MagicMock()
        mock_ollama.return_value = mock_llm

        # Mock agent creation and response
        mock_fallback_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.response = "Fallback analysis result"
        mock_fallback_agent.chat.return_value = mock_response
        mock_react_agent.return_value = mock_fallback_agent

        with patch("agents.agent_utils.settings", mock_settings):
            analyze_documents_agentic(
                None,  # No agent provided
                mock_index_data_complete,
                "Basic Analysis",
            )

            # Verify query engines were created
            mock_index_data_complete["vector"].as_query_engine.assert_called_once()
            mock_index_data_complete["kg"].as_query_engine.assert_called_once()

            # Verify QueryEngineTool creation
            assert mock_query_tool.call_count == 2

            # Verify tool metadata
            vector_call = mock_query_tool.call_args_list[0]
            vector_metadata = vector_call[1]["metadata"]
            assert vector_metadata.name == "vector_query"
            assert "vector similarity search" in vector_metadata.description

            kg_call = mock_query_tool.call_args_list[1]
            kg_metadata = kg_call[1]["metadata"]
            assert kg_metadata.name == "knowledge_graph_query"
            assert "knowledge graph" in kg_metadata.description

            # Verify Ollama LLM creation
            mock_ollama.assert_called_once_with(model=mock_settings.default_model)

            # Verify fallback agent creation
            mock_react_agent.assert_called_once_with(
                [mock_vector_tool, mock_kg_tool],
                llm=mock_llm,
                verbose=True,
            )

            # Verify analysis was performed
            mock_fallback_agent.chat.assert_called_once_with(
                "Analyze with prompt: Basic Analysis"
            )

            assert result == "Fallback analysis result"


class TestChatWithAgent:
    """Test async chat functionality with agents."""

    @pytest.mark.asyncio
    async def test_chat_with_agent_success(self, mock_agent):
        """Test successful async chat with agent."""
        # Mock memory buffer
        mock_memory = MagicMock(spec=ChatMemoryBuffer)

        # Mock async stream response
        async def mock_async_gen():
            for chunk in ["Hello ", "from ", "agent"]:
                yield chunk

        mock_async_response = MagicMock()
        mock_async_response.async_response_gen.return_value = mock_async_gen()
        mock_agent.async_stream_chat = AsyncMock(return_value=mock_async_response)

        # Collect response chunks
        response_chunks = []
        async for chunk in chat_with_agent(mock_agent, "Test query", mock_memory):
            response_chunks.append(chunk)

        # Verify agent async stream chat was called
        mock_agent.async_stream_chat.assert_called_once()

        # Verify response chunks
        assert response_chunks == ["Hello ", "from ", "agent"]

    @pytest.mark.asyncio
    async def test_chat_with_agent_error_handling(self, mock_agent):
        """Test error handling in async chat with agent."""
        # Mock memory buffer
        mock_memory = MagicMock(spec=ChatMemoryBuffer)

        # Mock async stream chat failure
        mock_agent.async_stream_chat = AsyncMock(
            side_effect=Exception("Chat generation failed")
        )

        with (
            patch("agents.agent_utils.logging.error") as mock_log_error,
            pytest.raises(Exception, match="Chat generation failed"),
        ):
            async for chunk in chat_with_agent(mock_agent, "Test query", mock_memory):
                pass  # Should not reach here

            # Verify error logging
            mock_log_error.assert_called_once()
            error_message = mock_log_error.call_args[0][0]
            assert "Chat generation error" in error_message

    @pytest.mark.asyncio
    async def test_chat_with_agent_empty_response(self, mock_agent):
        """Test async chat with empty response from agent."""
        # Mock memory buffer
        mock_memory = MagicMock(spec=ChatMemoryBuffer)

        # Mock empty async stream response
        async def mock_empty_gen():
            return
            yield  # unreachable

        mock_async_response = MagicMock()
        mock_async_response.async_response_gen.return_value = mock_empty_gen()
        mock_agent.async_stream_chat = AsyncMock(return_value=mock_async_response)

        # Collect response chunks
        response_chunks = []
        async for chunk in chat_with_agent(mock_agent, "Test query", mock_memory):
            response_chunks.append(chunk)

        # Verify no chunks received
        assert response_chunks == []

    @pytest.mark.asyncio
    async def test_chat_with_agent_multimodal_handling(self, mock_agent):
        """Test multimodal handling in async chat (Gemma/Nemotron support)."""
        # Mock memory buffer
        mock_memory = MagicMock(spec=ChatMemoryBuffer)

        # Mock async stream response with multimodal content
        async def mock_multimodal_gen():
            yield "Processing image... "
            yield "Analysis complete: "
            yield "The image shows technical diagrams."

        mock_async_response = MagicMock()
        mock_async_response.async_response_gen.return_value = mock_multimodal_gen()
        mock_agent.async_stream_chat = AsyncMock(return_value=mock_async_response)

        # Collect response chunks
        response_chunks = []
        async for chunk in chat_with_agent(
            mock_agent, "Analyze this image", mock_memory
        ):
            response_chunks.append(chunk)

        # Verify multimodal response processing
        assert len(response_chunks) == 3
        assert "Processing image" in response_chunks[0]
        assert "technical diagrams" in response_chunks[2]


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    @patch("agents.agent_utils.ColbertRerank")
    @patch("agents.agent_utils.RetrieverQueryEngine")
    @patch("agents.agent_utils.QueryEngineTool")
    @patch("agents.agent_utils.ReActAgent.from_tools")
    @patch("agents.agent_utils.ChatMemoryBuffer.from_defaults")
    def test_end_to_end_agent_creation_and_usage(
        self,
        mock_memory_buffer,
        mock_react_agent,
        mock_query_tool,
        mock_retriever_query_engine,
        mock_colbert_rerank,
        mock_index_data_complete,
        mock_llm,
        mock_settings,
    ):
        """Test end-to-end agent creation and tool usage."""
        # Mock reranker
        mock_reranker = MagicMock()
        mock_colbert_rerank.return_value = mock_reranker

        # Mock hybrid query engine
        mock_hybrid_engine = MagicMock()
        mock_retriever_query_engine.return_value = mock_hybrid_engine

        # Mock tools
        mock_hybrid_tool = MagicMock()
        mock_kg_tool = MagicMock()
        mock_hybrid_tool.metadata.name = "hybrid_fusion_search"
        mock_kg_tool.metadata.name = "knowledge_graph_query"
        mock_query_tool.side_effect = [mock_hybrid_tool, mock_kg_tool]

        # Mock memory and agent
        mock_memory = MagicMock()
        mock_memory_buffer.return_value = mock_memory
        mock_agent = MagicMock()
        mock_react_agent.return_value = mock_agent

        with patch("agents.agent_utils.settings", mock_settings):
            # Step 1: Create tools from index
            tools = create_tools_from_index(mock_index_data_complete)

            # Step 2: Create agent with tools
            agent = create_agent_with_tools(mock_index_data_complete, mock_llm)

            # Verify complete pipeline
            assert len(tools) == 2
            assert agent == mock_agent

            # Verify tool configuration
            mock_colbert_rerank.assert_called_once_with(
                model=mock_settings.reranker_model,
                top_n=mock_settings.reranking_top_k,
                keep_retrieval_score=True,
            )

            # Verify hybrid query engine configuration
            mock_retriever_query_engine.assert_called_once_with(
                retriever=mock_index_data_complete["retriever"],
                node_postprocessors=[mock_reranker],
            )

            # Verify agent configuration
            mock_react_agent.assert_called_once_with(
                tools=[mock_hybrid_tool, mock_kg_tool],
                llm=mock_llm,
                verbose=True,
                max_iterations=10,
                memory=mock_memory,
            )

    def test_graceful_degradation_scenario(self, mock_llm, mock_settings):
        """Test graceful degradation when components are missing."""
        # Create index data with missing components
        incomplete_index_data = {
            "vector": MagicMock(),
            "kg": None,  # Missing KG
            "retriever": None,  # Missing hybrid retriever
        }

        # Mock vector query engine
        mock_vector_engine = MagicMock()
        incomplete_index_data[
            "vector"
        ].as_query_engine.return_value = mock_vector_engine

        with patch("agents.agent_utils.settings", mock_settings):
            # Should handle missing components gracefully
            with patch("agents.agent_utils.QueryEngineTool") as mock_query_tool:
                with patch(
                    "agents.agent_utils.ReActAgent.from_tools"
                ) as mock_react_agent:
                    with patch(
                        "agents.agent_utils.logging.warning"
                    ) as mock_log_warning:
                        mock_vector_tool = MagicMock()
                        mock_query_tool.return_value = mock_vector_tool
                        mock_agent = MagicMock()
                        mock_react_agent.return_value = mock_agent

                        # Create tools (should fallback to vector only)
                        tools = create_tools_from_index(incomplete_index_data)

                        # Create agent (should work with single tool)
                        agent = create_agent_with_tools(incomplete_index_data, mock_llm)

                        # Verify graceful handling
                        assert len(tools) == 1  # Only vector tool
                        assert agent == mock_agent

                        # Verify warning about missing KG
                        mock_log_warning.assert_called()
                        warning_message = mock_log_warning.call_args[0][0]
                        assert "Knowledge Graph index not available" in warning_message

    @pytest.mark.asyncio
    async def test_performance_scenario_high_concurrency(self, mock_agent):
        """Test performance with high concurrency async operations."""
        # Mock memory buffer
        mock_memory = MagicMock(spec=ChatMemoryBuffer)

        # Mock async stream response
        async def mock_fast_gen(query_id: int):
            yield f"Response {query_id} chunk 1"
            yield f"Response {query_id} chunk 2"

        # Create multiple concurrent chat sessions
        async def mock_stream_chat(user_input, memory=None):
            query_id = int(user_input.split()[-1])  # Extract ID from query
            mock_async_response = MagicMock()
            mock_async_response.async_response_gen.return_value = mock_fast_gen(
                query_id
            )
            return mock_async_response

        mock_agent.async_stream_chat = mock_stream_chat

        # Run multiple concurrent chats
        async def single_chat(query_id: int) -> list[str]:
            chunks = []
            async for chunk in chat_with_agent(
                mock_agent, f"Query {query_id}", mock_memory
            ):
                chunks.append(chunk)
            return chunks

        # Execute 10 concurrent chats
        tasks = [single_chat(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all chats completed successfully
        assert len(results) == 10
        for i, result in enumerate(results):
            assert len(result) == 2  # Each chat should have 2 chunks
            assert f"Response {i}" in result[0]
            assert f"Response {i}" in result[1]


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    def test_invalid_index_data_structure(self, mock_settings):
        """Test handling of invalid index data structure."""
        invalid_index_data = "not a dictionary"

        with (
            patch("agents.agent_utils.settings", mock_settings),
            pytest.raises((AttributeError, TypeError)),
        ):
            create_tools_from_index(invalid_index_data)

    def test_colbert_rerank_initialization_failure(
        self, mock_index_data_complete, mock_settings
    ):
        """Test handling of ColBERT reranker initialization failure."""
        with patch("agents.agent_utils.ColbertRerank") as mock_colbert_rerank:
            # Mock reranker initialization failure
            mock_colbert_rerank.side_effect = Exception("Reranker init failed")

            with (
                patch("agents.agent_utils.settings", mock_settings),
                pytest.raises(Exception, match="Reranker init failed"),
            ):
                create_tools_from_index(mock_index_data_complete)

    @patch("agents.agent_utils.create_tools_from_index")
    def test_agent_creation_with_tool_creation_failure(
        self, mock_create_tools, mock_index_data_complete, mock_llm
    ):
        """Test agent creation when tool creation fails."""
        # Mock tool creation failure
        mock_create_tools.side_effect = Exception("Tool creation failed")

        with pytest.raises(Exception, match="Tool creation failed"):
            create_agent_with_tools(mock_index_data_complete, mock_llm)

    @pytest.mark.asyncio
    async def test_asyncio_to_thread_failure(self, mock_agent):
        """Test handling of asyncio.to_thread failure."""
        # Mock memory buffer
        mock_memory = MagicMock(spec=ChatMemoryBuffer)

        # Mock asyncio.to_thread failure
        with patch("agents.agent_utils.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = RuntimeError("Thread creation failed")

            with pytest.raises(RuntimeError, match="Thread creation failed"):
                async for chunk in chat_with_agent(
                    mock_agent, "Test query", mock_memory
                ):
                    pass  # Should not reach here
