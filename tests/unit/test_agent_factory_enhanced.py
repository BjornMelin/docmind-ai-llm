"""Enhanced comprehensive test coverage for agent_factory.py.

This test suite provides critical path coverage for the agent_factory module,
focusing on LangGraph supervisor patterns, multi-agent coordination,
query analysis, and routing logic to achieve 70%+ coverage.

Key areas covered:
- Query complexity analysis and classification
- Single and multi-agent system creation
- LangGraph supervisor routing logic
- Specialist agent creation (document, KG, multimodal)
- Agent system selection and fallback mechanisms
"""

from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.messages import HumanMessage
from llama_index.core.memory import ChatMemoryBuffer

# Import functions under test
from agent_factory import (
    AgentState,
    analyze_query_complexity,
    create_document_specialist_agent,
    create_knowledge_specialist_agent,
    create_langgraph_supervisor_system,
    create_multimodal_specialist_agent,
    create_single_agent,
    get_agent_system,
    process_query_with_agent_system,
    supervisor_routing_logic,
)


class TestQueryComplexityAnalysis:
    """Test query complexity analysis and classification."""

    def test_analyze_query_complexity_simple_queries(self):
        """Test classification of simple queries."""
        simple_queries = [
            "What is the summary?",
            "Show me the title.",
            "Give me info.",
            "Quick answer.",
        ]

        for query in simple_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert complexity == "simple"
            assert query_type in ["general", "document"]

    def test_analyze_query_complexity_moderate_queries(self):
        """Test classification of moderate complexity queries."""
        moderate_queries = [
            "How does this system work and what are the components?",
            "Compare the main features described in the document.",
            "What are the various approaches mentioned?",
        ]

        for query in moderate_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert complexity == "moderate"

    def test_analyze_query_complexity_complex_queries(self):
        """Test classification of complex queries."""
        complex_queries = [
            "Compare and analyze the relationship between multiple concepts across documents and explain the difference.",
            "How does the interaction between various systems work and why do they behave differently?",
            "Analyze the connections among several entities and summarize all findings.",
        ]

        for query in complex_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert complexity == "complex"

    def test_analyze_query_complexity_multimodal_queries(self):
        """Test classification of multimodal queries."""
        multimodal_queries = [
            "Show me the image in the document.",
            "What does the picture demonstrate?",
            "Analyze the visual content and diagram.",
            "Describe the chart in detail.",
        ]

        for query in multimodal_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert query_type == "multimodal"

    def test_analyze_query_complexity_knowledge_graph_queries(self):
        """Test classification of knowledge graph queries."""
        kg_queries = [
            "What entities are connected to this concept?",
            "Show me the relationship between these items.",
            "Which concepts are related to the main topic?",
            "Find entities connected to this subject.",
        ]

        for query in kg_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert query_type == "knowledge_graph"

    def test_analyze_query_complexity_document_queries(self):
        """Test classification of document-focused queries."""
        doc_queries = [
            "What does the document say about this topic?",
            "Find information in the text about X.",
            "Search the content for specific passage.",
            "What is mentioned in the document?",
        ]

        for query in doc_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert query_type == "document"

    def test_analyze_query_complexity_edge_cases(self):
        """Test edge cases in query analysis."""
        edge_cases = [
            "",  # Empty query
            "a",  # Single character
            "?",  # Just punctuation
            "   ",  # Whitespace only
        ]

        for query in edge_cases:
            complexity, query_type = analyze_query_complexity(query)
            # Should not crash and return valid values
            assert complexity in ["simple", "moderate", "complex"]
            assert query_type in [
                "general",
                "document",
                "knowledge_graph",
                "multimodal",
            ]

    def test_analyze_query_complexity_logging(self):
        """Test that query analysis includes logging."""
        with patch("agent_factory.logger") as mock_logger:
            analyze_query_complexity("test query for logging")

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0]
            assert "Query analysis:" in call_args[0]
            assert "complexity=" in call_args[0]
            assert "type=" in call_args[0]


class TestSingleAgentCreation:
    """Test single ReAct agent creation."""

    def test_create_single_agent_success(self):
        """Test successful single agent creation."""
        mock_tools = [MagicMock(), MagicMock()]
        mock_llm = MagicMock()

        with patch("agent_factory.ReActAgent.from_tools") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            with patch("agent_factory.logger") as mock_logger:
                result = create_single_agent(mock_tools, mock_llm)

                assert result == mock_agent

                # Verify configuration
                mock_agent_class.assert_called_once_with(
                    tools=mock_tools,
                    llm=mock_llm,
                    memory=pytest.mock.ANY,
                    verbose=True,
                    max_iterations=10,
                )

                # Verify memory configuration
                call_kwargs = mock_agent_class.call_args[1]
                memory = call_kwargs["memory"]
                assert isinstance(memory, ChatMemoryBuffer)

                # Verify success logging
                mock_logger.info.assert_called_once()

    def test_create_single_agent_with_custom_memory(self):
        """Test single agent creation with custom memory."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()
        custom_memory = MagicMock()

        with patch("agent_factory.ReActAgent.from_tools") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            result = create_single_agent(mock_tools, mock_llm, custom_memory)

            # Should use provided memory
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs["memory"] == custom_memory

    def test_create_single_agent_no_tools(self):
        """Test single agent creation with empty tools list."""
        mock_llm = MagicMock()

        with patch("agent_factory.ReActAgent.from_tools") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            result = create_single_agent([], mock_llm)

            assert result == mock_agent
            # Should still work with empty tools
            mock_agent_class.assert_called_once()

    def test_create_single_agent_creation_failure(self):
        """Test handling of agent creation failure."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()

        with patch(
            "agent_factory.ReActAgent.from_tools",
            side_effect=RuntimeError("Agent creation failed"),
        ):
            with patch("agent_factory.logger") as mock_logger:
                with pytest.raises(RuntimeError):
                    create_single_agent(mock_tools, mock_llm)

                # Should log the error
                mock_logger.error.assert_called_once()


class TestSpecialistAgentCreation:
    """Test specialist agent creation functions."""

    def test_create_document_specialist_agent_success(self):
        """Test successful document specialist agent creation."""
        mock_tools = [MagicMock(), MagicMock()]
        # Configure tool names
        mock_tools[0].metadata.name = "vector_search"
        mock_tools[1].metadata.name = "hybrid_search"

        mock_llm = MagicMock()

        with patch("agent_factory.create_react_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            result = create_document_specialist_agent(mock_tools, mock_llm)

            assert result == mock_agent

            # Verify agent configuration
            mock_create.assert_called_once_with(
                model=mock_llm,
                tools=pytest.mock.ANY,  # Filtered tools
                messages_modifier=pytest.mock.ANY,
            )

            # Verify message modifier contains document specialist context
            call_kwargs = mock_create.call_args[1]
            modifier = call_kwargs["messages_modifier"]
            assert "document processing specialist" in modifier.lower()
            assert "hybrid search" in modifier.lower()

    def test_create_document_specialist_agent_tool_filtering(self):
        """Test that document specialist filters for vector tools."""
        mock_tools = [MagicMock(), MagicMock(), MagicMock()]
        # Mix of vector and non-vector tools
        mock_tools[0].metadata.name = "vector_search"
        mock_tools[1].metadata.name = "knowledge_graph"
        mock_tools[2].metadata.name = "hybrid_vector_search"

        mock_llm = MagicMock()

        with patch("agent_factory.create_react_agent") as mock_create:
            create_document_specialist_agent(mock_tools, mock_llm)

            # Should only pass vector tools
            call_kwargs = mock_create.call_args[1]
            filtered_tools = call_kwargs["tools"]
            assert len(filtered_tools) == 2  # vector_search and hybrid_vector_search

    def test_create_knowledge_specialist_agent_success(self):
        """Test successful knowledge graph specialist agent creation."""
        mock_tools = [MagicMock(), MagicMock()]
        mock_tools[0].metadata.name = "knowledge_graph"
        mock_tools[1].metadata.name = "vector_search"

        mock_llm = MagicMock()

        with patch("agent_factory.create_react_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            result = create_knowledge_specialist_agent(mock_tools, mock_llm)

            assert result == mock_agent

            # Verify KG specialist context
            call_kwargs = mock_create.call_args[1]
            modifier = call_kwargs["messages_modifier"]
            assert "knowledge graph specialist" in modifier.lower()
            assert "entity relationships" in modifier.lower()

            # Should filter for knowledge tools
            filtered_tools = call_kwargs["tools"]
            assert len(filtered_tools) == 1  # Only knowledge_graph tool

    def test_create_multimodal_specialist_agent_success(self):
        """Test successful multimodal specialist agent creation."""
        mock_tools = [MagicMock(), MagicMock()]
        mock_llm = MagicMock()

        with patch("agent_factory.create_react_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            result = create_multimodal_specialist_agent(mock_tools, mock_llm)

            assert result == mock_agent

            # Verify multimodal specialist context
            call_kwargs = mock_create.call_args[1]
            modifier = call_kwargs["messages_modifier"]
            assert "multimodal content specialist" in modifier.lower()
            assert "text and image content" in modifier.lower()

            # Should use all tools for multimodal processing
            assert call_kwargs["tools"] == mock_tools

    def test_specialist_agents_with_empty_tools(self):
        """Test specialist agent creation with empty tools list."""
        mock_llm = MagicMock()

        with patch("agent_factory.create_react_agent") as mock_create:
            # All should handle empty tools gracefully
            create_document_specialist_agent([], mock_llm)
            create_knowledge_specialist_agent([], mock_llm)
            create_multimodal_specialist_agent([], mock_llm)

            assert mock_create.call_count == 3


class TestSupervisorRoutingLogic:
    """Test LangGraph supervisor routing logic."""

    def test_supervisor_routing_logic_no_messages(self):
        """Test routing when no messages in state."""
        state = AgentState()
        state["messages"] = []

        result = supervisor_routing_logic(state)

        # Should default to document specialist
        assert result == "document_specialist"

    def test_supervisor_routing_logic_multimodal_query(self):
        """Test routing for multimodal queries."""
        state = AgentState()
        state["messages"] = [HumanMessage(content="Show me the image in the document")]

        result = supervisor_routing_logic(state)

        assert result == "multimodal_specialist"
        assert state["query_type"] == "multimodal"

    def test_supervisor_routing_logic_knowledge_graph_query(self):
        """Test routing for knowledge graph queries."""
        state = AgentState()
        state["messages"] = [
            HumanMessage(content="What entities are connected to this concept?")
        ]

        result = supervisor_routing_logic(state)

        assert result == "knowledge_specialist"
        assert state["query_type"] == "knowledge_graph"

    def test_supervisor_routing_logic_complex_query(self):
        """Test routing for complex queries."""
        state = AgentState()
        state["messages"] = [
            HumanMessage(
                content="Compare and analyze the relationship between multiple concepts"
            )
        ]

        result = supervisor_routing_logic(state)

        assert result == "document_specialist"
        assert state["query_complexity"] == "complex"

    def test_supervisor_routing_logic_simple_query(self):
        """Test routing for simple queries."""
        state = AgentState()
        state["messages"] = [HumanMessage(content="What is the summary?")]

        result = supervisor_routing_logic(state)

        assert result == "document_specialist"
        assert state["query_complexity"] == "simple"

    def test_supervisor_routing_logic_string_message(self):
        """Test routing when message is string instead of HumanMessage."""
        state = AgentState()
        state["messages"] = ["Simple string query"]

        result = supervisor_routing_logic(state)

        assert result == "document_specialist"

    def test_supervisor_routing_logic_state_updates(self):
        """Test that routing logic updates state correctly."""
        state = AgentState()
        state["messages"] = [HumanMessage(content="Compare multiple systems")]

        supervisor_routing_logic(state)

        # State should be updated with analysis results
        assert "query_complexity" in state
        assert "query_type" in state
        assert state["query_complexity"] in ["simple", "moderate", "complex"]
        assert state["query_type"] in [
            "general",
            "document",
            "knowledge_graph",
            "multimodal",
        ]


class TestLangGraphSupervisorSystem:
    """Test LangGraph supervisor multi-agent system creation."""

    def test_create_langgraph_supervisor_system_success(self):
        """Test successful supervisor system creation."""
        mock_tools = [MagicMock(), MagicMock()]
        mock_llm = MagicMock()

        with patch("agent_factory.create_document_specialist_agent") as mock_doc:
            with patch("agent_factory.create_knowledge_specialist_agent") as mock_kg:
                with patch(
                    "agent_factory.create_multimodal_specialist_agent"
                ) as mock_mm:
                    with patch("agent_factory.StateGraph") as mock_graph_class:
                        mock_doc_agent = MagicMock()
                        mock_kg_agent = MagicMock()
                        mock_mm_agent = MagicMock()

                        mock_doc.return_value = mock_doc_agent
                        mock_kg.return_value = mock_kg_agent
                        mock_mm.return_value = mock_mm_agent

                        mock_graph = MagicMock()
                        mock_compiled_graph = MagicMock()
                        mock_graph.compile.return_value = mock_compiled_graph
                        mock_graph_class.return_value = mock_graph

                        with patch("agent_factory.logger") as mock_logger:
                            result = create_langgraph_supervisor_system(
                                mock_tools, mock_llm
                            )

                            assert result == mock_compiled_graph

                            # Verify graph construction
                            mock_graph_class.assert_called_once_with(AgentState)

                            # Verify agent nodes were added
                            expected_calls = [
                                call("document_specialist", mock_doc_agent),
                                call("knowledge_specialist", mock_kg_agent),
                                call("multimodal_specialist", mock_mm_agent),
                            ]
                            mock_graph.add_node.assert_has_calls(
                                expected_calls, any_order=True
                            )

                            # Verify conditional edges were added
                            mock_graph.add_conditional_edges.assert_called_once()

                            # Verify end edges were added
                            end_edge_calls = [
                                call("document_specialist", pytest.mock.ANY),
                                call("knowledge_specialist", pytest.mock.ANY),
                                call("multimodal_specialist", pytest.mock.ANY),
                            ]
                            mock_graph.add_edge.assert_has_calls(
                                end_edge_calls, any_order=True
                            )

                            # Verify success logging
                            mock_logger.info.assert_called_once()

    def test_create_langgraph_supervisor_system_failure(self):
        """Test handling of supervisor system creation failure."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()

        with patch(
            "agent_factory.create_document_specialist_agent",
            side_effect=RuntimeError("Agent creation failed"),
        ):
            with patch("agent_factory.logger") as mock_logger:
                result = create_langgraph_supervisor_system(mock_tools, mock_llm)

                assert result is None

                # Should log the error
                mock_logger.error.assert_called_once()

    def test_create_langgraph_supervisor_system_graph_compile_failure(self):
        """Test handling when graph compilation fails."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()

        with patch(
            "agent_factory.create_document_specialist_agent", return_value=MagicMock()
        ):
            with patch(
                "agent_factory.create_knowledge_specialist_agent",
                return_value=MagicMock(),
            ):
                with patch(
                    "agent_factory.create_multimodal_specialist_agent",
                    return_value=MagicMock(),
                ):
                    with patch("agent_factory.StateGraph") as mock_graph_class:
                        mock_graph = MagicMock()
                        mock_graph.compile.side_effect = RuntimeError("Compile failed")
                        mock_graph_class.return_value = mock_graph

                        with patch("agent_factory.logger") as mock_logger:
                            result = create_langgraph_supervisor_system(
                                mock_tools, mock_llm
                            )

                            assert result is None
                            mock_logger.error.assert_called_once()


class TestAgentSystemSelection:
    """Test agent system selection and configuration."""

    def test_get_agent_system_multi_agent_success(self):
        """Test getting multi-agent system when available."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()

        with patch(
            "agent_factory.create_langgraph_supervisor_system"
        ) as mock_create_multi:
            mock_multi_system = MagicMock()
            mock_create_multi.return_value = mock_multi_system

            agent_system, mode = get_agent_system(
                mock_tools, mock_llm, enable_multi_agent=True
            )

            assert agent_system == mock_multi_system
            assert mode == "multi"

            mock_create_multi.assert_called_once_with(mock_tools, mock_llm)

    def test_get_agent_system_multi_agent_fallback(self):
        """Test fallback to single agent when multi-agent fails."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()

        with patch(
            "agent_factory.create_langgraph_supervisor_system", return_value=None
        ):
            with patch("agent_factory.create_single_agent") as mock_create_single:
                mock_single_agent = MagicMock()
                mock_create_single.return_value = mock_single_agent

                with patch("agent_factory.logger") as mock_logger:
                    agent_system, mode = get_agent_system(
                        mock_tools, mock_llm, enable_multi_agent=True
                    )

                    assert agent_system == mock_single_agent
                    assert mode == "single"

                    # Should warn about fallback
                    mock_logger.warning.assert_called_once()

    def test_get_agent_system_single_agent_requested(self):
        """Test getting single agent when specifically requested."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()
        mock_memory = MagicMock()

        with patch("agent_factory.create_single_agent") as mock_create_single:
            mock_single_agent = MagicMock()
            mock_create_single.return_value = mock_single_agent

            agent_system, mode = get_agent_system(
                mock_tools, mock_llm, enable_multi_agent=False, memory=mock_memory
            )

            assert agent_system == mock_single_agent
            assert mode == "single"

            mock_create_single.assert_called_once_with(
                mock_tools, mock_llm, mock_memory
            )

    def test_get_agent_system_with_memory(self):
        """Test agent system creation with custom memory."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()
        custom_memory = MagicMock()

        with patch("agent_factory.create_single_agent") as mock_create_single:
            get_agent_system(mock_tools, mock_llm, memory=custom_memory)

            # Should pass memory to single agent creation
            mock_create_single.assert_called_once_with(
                mock_tools, mock_llm, custom_memory
            )


class TestQueryProcessing:
    """Test query processing with different agent systems."""

    def test_process_query_with_multi_agent_system(self):
        """Test processing query with multi-agent system."""
        mock_multi_system = MagicMock()

        # Mock LangGraph response
        mock_result = {"messages": [HumanMessage(content="Multi-agent response")]}
        mock_multi_system.invoke.return_value = mock_result

        result = process_query_with_agent_system(
            mock_multi_system, "test query", "multi"
        )

        assert result == "Multi-agent response"

        # Verify invoke was called with correct state
        call_args = mock_multi_system.invoke.call_args[0][0]
        assert isinstance(call_args, AgentState)
        assert len(call_args.messages) == 1
        assert call_args.messages[0].content == "test query"

    def test_process_query_with_single_agent_system(self):
        """Test processing query with single agent system."""
        mock_single_agent = MagicMock()
        mock_memory = MagicMock()

        # Mock agent chat response
        mock_response = MagicMock()
        mock_response.response = "Single agent response"
        mock_single_agent.chat.return_value = mock_response

        result = process_query_with_agent_system(
            mock_single_agent, "test query", "single", mock_memory
        )

        assert result == "Single agent response"
        mock_single_agent.chat.assert_called_once_with("test query")

    def test_process_query_multi_agent_no_messages(self):
        """Test multi-agent processing when no messages in result."""
        mock_multi_system = MagicMock()
        mock_multi_system.invoke.return_value = {"other_data": "value"}

        result = process_query_with_agent_system(
            mock_multi_system, "test query", "multi"
        )

        assert "Multi-agent processing completed but no response generated" in result

    def test_process_query_multi_agent_string_message(self):
        """Test multi-agent processing with string message."""
        mock_multi_system = MagicMock()
        mock_result = {"messages": ["String response instead of HumanMessage"]}
        mock_multi_system.invoke.return_value = mock_result

        result = process_query_with_agent_system(
            mock_multi_system, "test query", "multi"
        )

        assert result == "String response instead of HumanMessage"

    def test_process_query_single_agent_no_chat_method(self):
        """Test single agent processing when agent lacks chat method."""
        mock_single_agent = MagicMock()
        # Remove chat method
        del mock_single_agent.chat

        result = process_query_with_agent_system(
            mock_single_agent, "test query", "single"
        )

        assert result == "Single agent processing error."

    def test_process_query_exception_handling(self):
        """Test exception handling in query processing."""
        mock_agent_system = MagicMock()
        mock_agent_system.invoke.side_effect = RuntimeError("Processing failed")

        with patch("agent_factory.logger") as mock_logger:
            result = process_query_with_agent_system(
                mock_agent_system, "test query", "multi"
            )

            assert "Error processing query" in result
            assert "Processing failed" in result

            # Should log the error
            mock_logger.error.assert_called_once()

    def test_process_query_single_agent_chat_failure(self):
        """Test single agent processing when chat method fails."""
        mock_single_agent = MagicMock()
        mock_single_agent.chat.side_effect = RuntimeError("Chat failed")

        with patch("agent_factory.logger") as mock_logger:
            result = process_query_with_agent_system(
                mock_single_agent, "test query", "single"
            )

            assert "Error processing query" in result
            mock_logger.error.assert_called_once()


class TestAgentStateClass:
    """Test AgentState class functionality."""

    def test_agent_state_initialization(self):
        """Test AgentState initialization with default values."""
        state = AgentState()

        # Check default values
        assert state.query_complexity == "simple"
        assert state.query_type == "general"
        assert state.current_agent == "supervisor"
        assert state.task_progress == {}
        assert state.agent_outputs == {}
        assert state.final_answer == ""
        assert state.confidence_score == 0.0

    def test_agent_state_with_messages(self):
        """Test AgentState with messages from MessagesState."""
        messages = [HumanMessage(content="test")]
        state = AgentState(messages=messages)

        assert state.messages == messages
        # Default values should still be set
        assert state.query_complexity == "simple"

    def test_agent_state_custom_values(self):
        """Test AgentState with custom values."""
        state = AgentState(
            query_complexity="complex",
            query_type="multimodal",
            current_agent="document_specialist",
            final_answer="Custom answer",
            confidence_score=0.8,
        )

        assert state.query_complexity == "complex"
        assert state.query_type == "multimodal"
        assert state.current_agent == "document_specialist"
        assert state.final_answer == "Custom answer"
        assert state.confidence_score == 0.8

    def test_agent_state_task_progress_updates(self):
        """Test AgentState task progress tracking."""
        state = AgentState()

        # Update task progress
        state.task_progress["document_analysis"] = {
            "status": "completed",
            "results": ["item1", "item2"],
        }
        state.agent_outputs["document_specialist"] = "Analysis complete"

        assert state.task_progress["document_analysis"]["status"] == "completed"
        assert len(state.task_progress["document_analysis"]["results"]) == 2
        assert state.agent_outputs["document_specialist"] == "Analysis complete"


class TestAgentFactoryIntegration:
    """Test integration scenarios in agent factory."""

    def test_complete_multi_agent_workflow(self):
        """Test complete multi-agent workflow from query to response."""
        mock_tools = [MagicMock(), MagicMock()]
        mock_llm = MagicMock()

        with patch(
            "agent_factory.create_langgraph_supervisor_system"
        ) as mock_create_multi:
            mock_multi_system = MagicMock()
            mock_result = {
                "messages": [HumanMessage(content="Multi-agent analysis complete")]
            }
            mock_multi_system.invoke.return_value = mock_result
            mock_create_multi.return_value = mock_multi_system

            # Step 1: Get multi-agent system
            agent_system, mode = get_agent_system(
                mock_tools, mock_llm, enable_multi_agent=True
            )

            assert mode == "multi"

            # Step 2: Process query
            result = process_query_with_agent_system(
                agent_system, "Analyze the relationship between entities", mode
            )

            assert result == "Multi-agent analysis complete"

    def test_single_agent_fallback_workflow(self):
        """Test complete fallback workflow to single agent."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()

        # Multi-agent creation fails
        with patch(
            "agent_factory.create_langgraph_supervisor_system", return_value=None
        ):
            with patch("agent_factory.create_single_agent") as mock_create_single:
                mock_single_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.response = "Single agent fallback response"
                mock_single_agent.chat.return_value = mock_response
                mock_create_single.return_value = mock_single_agent

                with patch("agent_factory.logger"):
                    # Step 1: Get agent system (should fallback)
                    agent_system, mode = get_agent_system(
                        mock_tools, mock_llm, enable_multi_agent=True
                    )

                    assert mode == "single"

                    # Step 2: Process query
                    result = process_query_with_agent_system(
                        agent_system, "Simple query", mode
                    )

                    assert result == "Single agent fallback response"

    def test_query_analysis_to_routing_integration(self):
        """Test integration between query analysis and routing."""
        # Test multimodal query routing
        state = AgentState()
        state["messages"] = [HumanMessage(content="Show me the image content")]

        route = supervisor_routing_logic(state)

        assert route == "multimodal_specialist"
        assert state["query_type"] == "multimodal"

        # Test KG query routing
        state = AgentState()
        state["messages"] = [HumanMessage(content="What entities are connected?")]

        route = supervisor_routing_logic(state)

        assert route == "knowledge_specialist"
        assert state["query_type"] == "knowledge_graph"
