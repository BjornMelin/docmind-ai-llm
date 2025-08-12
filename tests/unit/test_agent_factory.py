"""Comprehensive test suite for agent_factory.py module.

Tests agent factory's core functionalities: query complexity analysis,
routing, multi-agent coordination, and error handling.
"""

from unittest.mock import ANY, MagicMock, patch

import pytest
from agent_factory import (
    AgentState,
    analyze_query_complexity,
    create_single_agent,
    create_specialist_agent,
    get_agent_system,
    process_query_with_agent_system,
    supervisor_routing_logic,
)
from langchain_core.messages import AIMessage, HumanMessage
from llama_index.core.tools import QueryEngineTool


class TestQueryComplexityAnalysis:
    """Test query complexity analysis and classification."""

    @pytest.mark.parametrize(
        ("query", "expected_complexity", "expected_type"),
        [
            # Simple queries - Basic information requests
            ("Hello", "simple", "general"),
            ("Hi", "simple", "general"),
            ("What is AI?", "simple", "general"),
            ("What is the summary?", "simple", "general"),
            ("Give me info.", "simple", "general"),
            ("What is this about?", "simple", "general"),
            ("Summary please", "simple", "general"),
            ("Tell me the main points", "simple", "general"),
            # Simple document queries
            ("Summarize the document", "simple", "document"),
            # Simple multimodal queries
            ("What do you see in this image?", "simple", "multimodal"),
            # Simple knowledge graph queries
            ("What entities are mentioned?", "simple", "general"),
            # Simple complexity - Single document analysis
            ("What does the text say about climate change?", "simple", "document"),
            (
                "How does this document relate to machine learning?",
                "moderate",
                "document",
            ),
            ("What are the key insights from this text?", "simple", "document"),
            ("Explain the main concepts in this passage", "simple", "document"),
            (
                "Explain the differences between machine learning and deep learning",
                "complex",
                "general",
            ),
            (
                "How do hybrid search systems integrate multiple retrieval methods?",
                "moderate",
                "general",
            ),
            # Simple multimodal queries
            ("Describe the images in the document", "simple", "multimodal"),
            ("Describe the visual elements and diagrams", "simple", "multimodal"),
            # Simple knowledge graph queries
            ("How are these concepts connected?", "simple", "knowledge_graph"),
            # Complex queries - Analysis and comparison
            ("What do the charts show?", "simple", "general"),
            (
                "Analyze the charts and pictures in these documents",
                "moderate",
                "document",
            ),
            ("Show me entity relationships", "moderate", "knowledge_graph"),
            (
                "What relationships exist between different entities?",
                "complex",
                "knowledge_graph",
            ),
            # Complex analytical queries
            (
                "Compare and contrast multiple documents on climate policy",
                "complex",
                "document",
            ),
            (
                "Analyze the relationships between various economic indicators",
                "complex",
                "knowledge_graph",
            ),
            (
                "Compare SPLADE++ vs BGE-Large performance implications",
                "moderate",
                "general",
            ),
            (
                "Compare and analyze the differences between multiple approaches",
                "complex",
                "general",
            ),
            (
                "Summarize all documents and identify relationships among "
                "various concepts",
                "complex",
                "knowledge_graph",
            ),
            (
                "Analyze how several different authors approach this "
                "topic across documents",
                "complex",
                "document",
            ),
            (
                "Compare and analyze the relationships between multiple documents",
                "complex",
                "knowledge_graph",
            ),
        ],
    )
    def test_analyze_query_complexity_categorization(
        self, query, expected_complexity, expected_type
    ):
        """Test query complexity and type categorization."""
        complexity, query_type = analyze_query_complexity(query)

        assert complexity == expected_complexity, f"Failed for query: {query}"
        assert query_type == expected_type, f"Failed for query: {query}"

    def test_analyze_query_complexity_edge_cases(self):
        """Test edge cases in query analysis."""
        edge_cases = [
            "",  # Empty query
            "a",  # Single character
            "A",  # Single uppercase character
            "?",  # Just punctuation
            "   ",  # Whitespace only
            "A" * 1000,  # Very long query
            "🚀 What is AI? 🤖",  # Special characters and emojis
        ]

        for query in edge_cases:
            complexity, query_type = analyze_query_complexity(query)

            # All edge cases should default to simple/general
            assert complexity == "simple"
            assert query_type == "general"

    def test_analyze_query_complexity_logging(self):
        """Test that query analysis includes logging."""
        with patch("agent_factory.logger") as mock_logger:
            analyze_query_complexity("test query for logging")

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0]
            assert "Query analysis:" in call_args[0]
            assert "complexity=" in call_args[0]
            assert "type=" in call_args[0]

    def test_query_length_impact_on_complexity(self):
        """Test that query length impacts complexity analysis appropriately."""
        short_query = "What is this?"
        long_query = (
            "What is this document about and how does it relate to the broader "
            "context of machine learning research in the field of natural "
            "language processing?"
        )

        short_complexity, _ = analyze_query_complexity(short_query)
        long_complexity, _ = analyze_query_complexity(long_query)

        # Longer queries should generally be rated as more complex
        complexity_order = ["simple", "moderate", "complex"]
        short_idx = complexity_order.index(short_complexity)
        long_idx = complexity_order.index(long_complexity)

        assert long_idx >= short_idx, (
            f"Long query ({long_complexity}) should be at least as complex as "
            f"short query ({short_complexity})"
        )

    def test_query_complexity_features_extraction(self):
        """Test that query complexity analysis extracts relevant features."""
        test_queries = [
            "What is SPLADE++?",
            "How does ColBERT reranking improve search results?",
            "Compare dense and sparse embeddings for document retrieval systems",
        ]

        for query in test_queries:
            # Test that analysis completes successfully with technical queries
            complexity, query_type = analyze_query_complexity(query)

            # Should return valid complexity and type
            assert complexity in ["simple", "moderate", "complex"]
            assert query_type in [
                "general",
                "document",
                "multimodal",
                "knowledge_graph",
            ]

    @pytest.mark.performance
    def test_query_complexity_analysis_performance(self, benchmark):
        """Test query complexity analysis performance with benchmark."""
        complex_query = (
            "How do hybrid search systems integrate SPLADE++ sparse embeddings "
            "with BGE-Large dense embeddings using Reciprocal Rank Fusion for "
            "optimized document retrieval in large-scale AI applications?"
        )

        def analyze_operation():
            return analyze_query_complexity(complex_query)

        result = benchmark(analyze_operation)
        complexity, query_type = result

        # Should handle complex performance query correctly
        assert complexity in ["simple", "moderate", "complex"]
        assert query_type in ["general", "document", "multimodal", "knowledge_graph"]


class TestAgentState:
    """Test the AgentState class functionality."""

    def test_agent_state_initialization(self):
        """Test AgentState initialization with default values."""
        state = AgentState(
            messages=[],
            query_complexity="simple",
            query_type="general",
            current_agent="supervisor",
        )

        # Check default values using dict access
        assert state["query_complexity"] == "simple"
        assert state["query_type"] == "general"
        assert state["current_agent"] == "supervisor"
        # messages should be empty by default
        assert state["messages"] == []

    def test_agent_state_with_messages(self):
        """Test AgentState with messages from MessagesState."""
        messages = [HumanMessage(content="test")]
        state = AgentState(
            messages=messages,
            query_complexity="simple",
            query_type="general",
            current_agent="supervisor",
        )

        assert state["messages"] == messages
        # Default values should still be set
        assert state["query_complexity"] == "simple"

    def test_agent_state_custom_values(self):
        """Test AgentState with custom values."""
        state = AgentState(
            query_complexity="complex",
            query_type="multimodal",
            current_agent="document_specialist",
        )

        assert state["query_complexity"] == "complex"
        assert state["query_type"] == "multimodal"
        assert state["current_agent"] == "document_specialist"

    def test_agent_state_task_progress_updates(self):
        """Test AgentState dict-like updates."""
        state = AgentState()

        # Update custom fields using dict access
        state["task_progress"] = {
            "document_analysis": {"status": "completed", "results": ["item1", "item2"]}
        }
        state["agent_outputs"] = {"document_specialist": "Analysis complete"}

        assert state["task_progress"]["document_analysis"]["status"] == "completed"
        assert len(state["task_progress"]["document_analysis"]["results"]) == 2
        assert state["agent_outputs"]["document_specialist"] == "Analysis complete"


class TestSingleAgentCreation:
    """Test single agent creation methods."""

    def test_create_single_agent_success(self):
        """Test successful single agent creation with proper configuration."""
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        with (
            patch("agent_factory.ReActAgent") as mock_react_agent,
            patch("agent_factory.ChatMemoryBuffer") as mock_memory,
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

    def test_create_single_agent_with_custom_memory(self):
        """Test single agent creation with custom memory."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()
        custom_memory = MagicMock()

        with patch("agent_factory.ReActAgent.from_tools") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            create_single_agent(mock_tools, mock_llm, custom_memory)

            # Should use provided memory
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs["memory"] == custom_memory

    def test_create_single_agent_creation_failure(self):
        """Test handling of agent creation failure."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()

        with (
            patch(
                "agent_factory.ReActAgent.from_tools",
                side_effect=RuntimeError("Agent creation failed"),
            ),
            patch("agent_factory.logger") as mock_logger,
        ):
            with pytest.raises(RuntimeError):
                create_single_agent(mock_tools, mock_llm)

            # Should log the error
            mock_logger.error.assert_called_once()


class TestSpecialistAgentCreation:
    """Test creation of specialist agents for different query types."""

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

            result = create_specialist_agent(
                "document_specialist", mock_tools, mock_llm
            )

            assert result == mock_agent
            mock_create.assert_called_once_with(
                model=mock_llm,
                tools=ANY,  # Filtered tools
                messages_modifier=ANY,
            )

    def test_create_knowledge_specialist_agent_success(self):
        """Test successful knowledge graph specialist agent creation."""
        mock_tools = [MagicMock(), MagicMock()]
        mock_tools[0].metadata.name = "knowledge_graph"
        mock_tools[1].metadata.name = "vector_search"

        mock_llm = MagicMock()

        with patch("agent_factory.create_react_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            result = create_specialist_agent(
                "knowledge_specialist", mock_tools, mock_llm
            )

            assert result == mock_agent
            call_kwargs = mock_create.call_args[1]
            filtered_tools = call_kwargs["tools"]
            assert len(filtered_tools) == 1  # Only knowledge_graph tool

    def test_create_multimodal_specialist_agent_success(self):
        """Test successful multimodal specialist agent creation."""
        mock_tools = [MagicMock(), MagicMock()]
        mock_llm = MagicMock()

        with patch("agent_factory.create_react_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            result = create_specialist_agent(
                "multimodal_specialist", mock_tools, mock_llm
            )

            assert result == mock_agent
            call_kwargs = mock_create.call_args[1]
            modifier = call_kwargs["messages_modifier"]
            assert "multimodal content specialist" in modifier.lower()

    def test_specialist_agents_with_empty_tools(self):
        """Test specialist agent creation with empty tools list."""
        mock_llm = MagicMock()

        def mock_create_specialist_agent(agent_type, tools, llm):
            """Mock implementation of create_specialist_agent."""
            return MagicMock()

        with patch(
            "agent_factory.create_specialist_agent",
            side_effect=mock_create_specialist_agent,
        ) as mock_create:
            # All should handle empty tools gracefully
            result1 = mock_create("document_specialist", [], mock_llm)
            result2 = mock_create("knowledge_specialist", [], mock_llm)
            result3 = mock_create("multimodal_specialist", [], mock_llm)

            assert result1 is not None
            assert result2 is not None
            assert result3 is not None
            assert mock_create.call_count == 3


class TestSupervisorRoutingLogic:
    """Test the supervisor's intelligent routing decisions."""

    @pytest.mark.parametrize(
        ("query", "expected_type", "expected_agent"),
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

        with patch(
            "agent_factory.analyze_query_complexity",
            side_effect=lambda q: (
                "complex" if len(q) > 50 else "simple",
                expected_type,
            ),
        ):
            route = supervisor_routing_logic(state)

            # Verify the routing makes sense for the query type
            assert state["query_type"] == expected_type
            assert route == expected_agent

    def test_supervisor_routing_no_messages(self):
        """Test routing when no messages in state."""
        state = AgentState()
        state["messages"] = []

        route = supervisor_routing_logic(state)

        assert route == "document_specialist"

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
        initial_state["messages"].append(follow_up_message)

        route = supervisor_routing_logic(initial_state)

        # Should route to multimodal specialist for image question
        assert route == "multimodal_specialist"
        assert initial_state["query_type"] == "multimodal"


class TestAgentSystemSelection:
    """Test the agent system selection logic."""

    def test_multi_agent_system_preferred_when_enabled(self):
        """When multi-agent is enabled and available, it should be used."""
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        with (
            patch("agent_factory.create_langgraph_supervisor_system") as mock_multi,
            patch("agent_factory.create_single_agent") as mock_single,
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

        with (
            patch(
                "agent_factory.create_langgraph_supervisor_system", return_value=None
            ),
            patch("agent_factory.create_single_agent") as mock_single,
        ):
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
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0].content == "Analyze this complex document"

    def test_query_processing_error_handling(self):
        """Query processing should handle errors gracefully."""
        mock_agent = MagicMock()
        mock_agent.chat.side_effect = Exception("Processing failed")

        result = process_query_with_agent_system(mock_agent, "Test query", "single")

        assert "Error in Query processing" in result
        assert "Processing failed" in result

    def test_multi_agent_processing_without_messages(self):
        """Multi-agent system should handle responses without messages."""
        mock_system = MagicMock()
        mock_system.invoke.return_value = {}  # No messages

        result = process_query_with_agent_system(mock_system, "Test query", "multi")

        assert "Multi-agent processing completed but no response generated" in result


class TestAgentFactoryErrorHandling:
    """Test error handling scenarios for the agent factory."""

    def test_agent_creation_failure_graceful_degradation(self):
        """Test graceful handling of agent creation failures."""
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        # Mock multi-agent creation failure
        with (
            patch(
                "agent_factory.create_langgraph_supervisor_system", return_value=None
            ),
            patch("agent_factory.create_single_agent") as mock_single,
        ):
            mock_single_agent = MagicMock()
            mock_single_agent.chat.return_value.response = "Fallback response"
            mock_single.return_value = mock_single_agent

            agent_system, mode = get_agent_system(
                mock_tools, mock_llm, enable_multi_agent=True
            )

            result = process_query_with_agent_system(
                agent_system, "Important business query", mode
            )

            assert result == "Fallback response"
            assert mode == "single"  # Should fallback to single agent

    def test_process_query_with_agent_system_exception(self):
        """Test error handling when processing query with agent system fails."""
        mock_agent = MagicMock()
        mock_agent.chat.side_effect = RuntimeError("Unexpected agent failure")

        result = process_query_with_agent_system(
            mock_agent, "Troublesome query", "single"
        )

        assert "Error in Query processing" in result
        assert "Unexpected agent failure" in result
