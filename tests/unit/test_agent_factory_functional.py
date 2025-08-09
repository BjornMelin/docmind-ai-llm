"""Functional tests for agent_factory.py module.

This test suite validates the real business functionality of the agent factory,
focusing on user scenarios, query routing intelligence, and multi-agent coordination.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from llama_index.core.tools import QueryEngineTool

from agent_factory import (
    AgentState,
    analyze_query_complexity,
    create_langgraph_supervisor_system,
    create_single_agent,
    get_agent_system,
    process_query_with_agent_system,
    supervisor_routing_logic,
)


class TestQueryAnalysisIntelligence:
    """Test the intelligent query analysis system that drives agent routing."""

    def test_simple_greeting_queries_routed_correctly(self):
        """Users expect simple greetings to be handled efficiently."""
        simple_queries = [
            "Hello",
            "Hi there",
            "Good morning",
            "How are you?",
            "What can you do?",
        ]

        for query in simple_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert complexity == "simple", (
                f"Expected simple for '{query}', got {complexity}"
            )
            assert query_type == "general", (
                f"Expected general for '{query}', got {query_type}"
            )

    def test_document_analysis_queries_detected(self):
        """Users asking about document content should be routed to document specialists."""
        document_queries = [
            "Summarize this document",
            "What does the text say about climate change?",
            "Find key passages about artificial intelligence",
            "Extract the main points from the content",
        ]

        for query in document_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert query_type == "document", (
                f"Expected document type for '{query}', got {query_type}"
            )

    def test_multimodal_queries_detected(self):
        """Users asking about visual content should be routed to multimodal specialists."""
        multimodal_queries = [
            "Describe the images in the document",
            "What do the charts show?",
            "Analyze the diagrams and pictures",
            "Explain the visual content",
        ]

        for query in multimodal_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert query_type == "multimodal", (
                f"Expected multimodal for '{query}', got {query_type}"
            )

    def test_knowledge_graph_queries_detected(self):
        """Users asking about relationships should be routed to KG specialists."""
        kg_queries = [
            "Show me entity relationships",
            "How are these concepts connected?",
            "What entities are related to each other?",
            "Map the connections between topics",
        ]

        for query in kg_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert query_type == "knowledge_graph", (
                f"Expected knowledge_graph for '{query}', got {query_type}"
            )

    def test_complex_analytical_queries_detected(self):
        """Users making complex analytical requests should trigger appropriate complexity."""
        complex_queries = [
            "Compare and contrast multiple documents on climate policy",
            "Analyze the relationships between various economic indicators",
            "Provide comprehensive analysis across several research papers",
        ]

        for query in complex_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert complexity in ["complex", "moderate"], (
                f"Expected complex/moderate for '{query}', got {complexity}"
            )


class TestAgentStateManagement:
    """Test the agent state management for multi-agent coordination."""

    def test_agent_state_tracks_conversation_context(self):
        """Agent state should maintain context throughout multi-turn conversations."""
        state = AgentState()

        # Initial state
        assert state.query_complexity == "simple"
        assert state.current_agent == "supervisor"
        assert len(state.task_progress) == 0

        # State evolution during conversation
        state.query_complexity = "complex"
        state.current_agent = "document_specialist"
        state.task_progress["analysis"] = "in_progress"
        state.agent_outputs["summary"] = "Initial findings..."

        assert state.query_complexity == "complex"
        assert state.current_agent == "document_specialist"
        assert state.task_progress["analysis"] == "in_progress"
        assert "summary" in state.agent_outputs

    def test_agent_state_inherits_from_messages_state(self):
        """Agent state should properly handle LangGraph message passing."""
        state = AgentState()

        # Should support message handling
        assert hasattr(state, "messages")

        # Should be able to add messages
        test_message = HumanMessage(content="Test query")
        state.messages = [test_message]

        assert len(state.messages) == 1
        assert state.messages[0].content == "Test query"


class TestSupervisorRoutingLogic:
    """Test the supervisor's intelligent routing decisions."""

    def test_supervisor_routes_document_queries_appropriately(self):
        """Document queries should be routed to document specialist."""
        state = AgentState(
            messages=[HumanMessage(content="Summarize the research paper")]
        )

        with patch(
            "agent_factory.analyze_query_complexity",
            return_value=("moderate", "document"),
        ):
            route = supervisor_routing_logic(state)

            assert route == "document_specialist"
            assert state.query_complexity == "moderate"
            assert state.query_type == "document"

    def test_supervisor_routes_multimodal_queries_appropriately(self):
        """Multimodal queries should be routed to multimodal specialist."""
        state = AgentState(
            messages=[HumanMessage(content="Describe the images in the PDF")]
        )

        with patch(
            "agent_factory.analyze_query_complexity",
            return_value=("complex", "multimodal"),
        ):
            route = supervisor_routing_logic(state)

            assert route == "multimodal_specialist"
            assert state.query_type == "multimodal"

    def test_supervisor_routes_knowledge_graph_queries_appropriately(self):
        """Knowledge graph queries should be routed to KG specialist."""
        state = AgentState(messages=[HumanMessage(content="Show entity relationships")])

        with patch(
            "agent_factory.analyze_query_complexity",
            return_value=("complex", "knowledge_graph"),
        ):
            route = supervisor_routing_logic(state)

            assert route == "knowledge_specialist"
            assert state.query_type == "knowledge_graph"

    def test_supervisor_handles_empty_messages_gracefully(self):
        """Supervisor should handle edge cases like empty message lists."""
        state = AgentState(messages=[])

        route = supervisor_routing_logic(state)

        # Should default to a sensible route
        assert route == "document_specialist"


class TestSingleAgentCreation:
    """Test single agent creation for simple queries."""

    def test_single_agent_created_with_proper_configuration(self):
        """Single agents should be configured correctly for their task."""
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        with (
            patch("llama_index.core.agent.ReActAgent") as mock_react_agent,
            patch("llama_index.core.memory.ChatMemoryBuffer") as mock_memory
        ):
                mock_agent = MagicMock()
                mock_react_agent.from_tools.return_value = mock_agent
                mock_memory.from_defaults.return_value = MagicMock()

                result = create_single_agent(mock_tools, mock_llm)

                assert result == mock_agent
                # Verify proper configuration
                call_args = mock_react_agent.from_tools.call_args
                assert call_args[1]["tools"] == mock_tools
                assert call_args[1]["llm"] == mock_llm
                assert call_args[1]["verbose"] is True
                assert call_args[1]["max_iterations"] == 10

    def test_single_agent_handles_creation_failure(self):
        """Single agent creation should handle and propagate errors properly."""
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        with patch(
            "llama_index.core.agent.ReActAgent.from_tools",
            side_effect=RuntimeError("Agent creation failed"),
        ):
            with (
                patch("llama_index.core.memory.ChatMemoryBuffer"),
                pytest.raises(RuntimeError, match="Agent creation failed")
            ):
                    create_single_agent(mock_tools, mock_llm)


class TestMultiAgentSystemCreation:
    """Test multi-agent system creation and coordination."""

    def test_multi_agent_system_creates_specialist_agents(self):
        """Multi-agent system should create and configure specialist agents."""
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        with (
            patch("langgraph.graph.StateGraph") as mock_state_graph,
            patch("langgraph.prebuilt.create_react_agent") as mock_create_react
        ):
                # Setup mocks
                mock_graph = MagicMock()
                mock_state_graph.return_value = mock_graph
                mock_graph.compile.return_value = MagicMock()

                mock_agent = MagicMock()
                mock_create_react.return_value = mock_agent

                result = create_langgraph_supervisor_system(mock_tools, mock_llm)

                assert result is not None
                # Should create three specialist agents
                assert mock_create_react.call_count == 3
                # Should compile the graph
                mock_graph.compile.assert_called_once()

    def test_multi_agent_system_handles_creation_failure(self):
        """Multi-agent system should handle creation failures gracefully."""
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        with patch(
            "langgraph.graph.StateGraph", side_effect=Exception("Graph creation failed")
        ):
            result = create_langgraph_supervisor_system(mock_tools, mock_llm)

            assert result is None  # Should return None on failure


class TestAgentSystemSelection:
    """Test the agent system selection logic."""

    def test_multi_agent_system_preferred_when_enabled(self):
        """When multi-agent is enabled and available, it should be used."""
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        with (
            patch("agent_factory.create_langgraph_supervisor_system") as mock_multi,
            patch("agent_factory.create_single_agent") as mock_single
        ):
                mock_multi_system = MagicMock()
                mock_multi.return_value = mock_multi_system

                agent_system, mode = get_agent_system(
                    mock_tools, mock_llm, enable_multi_agent=True
                )

                assert agent_system == mock_multi_system
                assert mode == "multi"
                # Single agent should not be called
                mock_single.assert_not_called()

    def test_single_agent_fallback_when_multi_agent_fails(self):
        """Should fall back to single agent when multi-agent creation fails."""
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        with patch(
            "agent_factory.create_langgraph_supervisor_system", return_value=None
        ):
            with patch("agent_factory.create_single_agent") as mock_single:
                mock_single_agent = MagicMock()
                mock_single.return_value = mock_single_agent

                agent_system, mode = get_agent_system(
                    mock_tools, mock_llm, enable_multi_agent=True
                )

                assert agent_system == mock_single_agent
                assert mode == "single"

    def test_single_agent_used_when_multi_agent_disabled(self):
        """Single agent should be used when multi-agent is disabled."""
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        with patch("agent_factory.create_single_agent") as mock_single:
            mock_single_agent = MagicMock()
            mock_single.return_value = mock_single_agent

            agent_system, mode = get_agent_system(
                mock_tools, mock_llm, enable_multi_agent=False
            )

            assert agent_system == mock_single_agent
            assert mode == "single"


class TestQueryProcessing:
    """Test end-to-end query processing with different agent systems."""

    def test_single_agent_query_processing(self):
        """Single agent should process queries and return responses."""
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.response = "This is the agent's response"
        mock_agent.chat.return_value = mock_response

        result = process_query_with_agent_system(mock_agent, "What is AI?", "single")

        assert result == "This is the agent's response"
        mock_agent.chat.assert_called_once_with("What is AI?")

    def test_multi_agent_query_processing(self):
        """Multi-agent system should process queries through LangGraph."""
        mock_system = MagicMock()
        final_message = AIMessage(content="Multi-agent response")
        mock_system.invoke.return_value = {
            "messages": [final_message],
            "final_answer": "Multi-agent response",
        }

        result = process_query_with_agent_system(
            mock_system, "Analyze this complex document", "multi"
        )

        assert result == "Multi-agent response"
        # Verify proper state initialization
        call_args = mock_system.invoke.call_args[0][0]
        assert len(call_args.messages) == 1
        assert call_args.messages[0].content == "Analyze this complex document"

    def test_query_processing_error_handling(self):
        """Query processing should handle errors gracefully."""
        mock_agent = MagicMock()
        mock_agent.chat.side_effect = Exception("Processing failed")

        result = process_query_with_agent_system(mock_agent, "Test query", "single")

        assert "Error processing query" in result
        assert "Processing failed" in result

    def test_multi_agent_processing_without_messages(self):
        """Multi-agent system should handle responses without messages."""
        mock_system = MagicMock()
        mock_system.invoke.return_value = {}  # No messages

        result = process_query_with_agent_system(mock_system, "Test query", "multi")

        assert "Multi-agent processing completed but no response generated" in result


class TestRealWorldScenarios:
    """Test realistic user scenarios that would occur in production."""

    @pytest.mark.parametrize(
        "query,expected_type,expected_agent",
        [
            (
                "Help me understand this research paper",
                "document",
                "document_specialist",
            ),
            (
                "Show me the relationships between these entities",
                "knowledge_graph",
                "knowledge_specialist",
            ),
            (
                "Describe the charts in this report",
                "multimodal",
                "multimodal_specialist",
            ),
            (
                "What is machine learning?",
                "general",
                "document_specialist",
            ),  # Default routing
        ],
    )
    def test_realistic_query_routing_scenarios(
        self, query, expected_type, expected_agent
    ):
        """Test that realistic user queries are routed to appropriate specialists."""
        state = AgentState(messages=[HumanMessage(content=query)])

        route = supervisor_routing_logic(state)

        # Verify the routing makes sense for the query type
        assert state.query_type == expected_type
        assert route == expected_agent

    def test_conversation_context_maintained(self):
        """Multi-turn conversations should maintain context across interactions."""
        # Simulate a conversation flow
        initial_state = AgentState(
            messages=[HumanMessage(content="Analyze this document")],
            query_complexity="moderate",
            current_agent="document_specialist",
        )

        # Add follow-up question
        follow_up_message = HumanMessage(content="What about the images?")
        initial_state.messages.append(follow_up_message)

        route = supervisor_routing_logic(initial_state)

        # Should route to multimodal specialist for image question
        assert route == "multimodal_specialist"
        assert initial_state.query_type == "multimodal"

    def test_error_recovery_in_production_scenario(self):
        """System should recover gracefully from agent failures in production."""
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        # Simulate multi-agent creation failure
        with patch(
            "agent_factory.create_langgraph_supervisor_system", return_value=None
        ):
            with patch("agent_factory.create_single_agent") as mock_single:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.response = "Fallback response"
                mock_agent.chat.return_value = mock_response
                mock_single.return_value = mock_agent

                # Get fallback system
                agent_system, mode = get_agent_system(
                    mock_tools, mock_llm, enable_multi_agent=True
                )

                # Process query with fallback
                result = process_query_with_agent_system(
                    agent_system, "Important business query", mode
                )

                assert result == "Fallback response"
                assert mode == "single"  # Should fallback to single agent
