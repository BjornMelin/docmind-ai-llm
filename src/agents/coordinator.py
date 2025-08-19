"""Multi-Agent Coordination System using LangGraph supervisor pattern.

This module implements the main MultiAgentCoordinator that orchestrates five
specialized agents using LangGraph's supervisor architecture. The system
provides intelligent query processing with automatic routing, planning,
retrieval, synthesis, and validation.

Features:
- LangGraph supervisor with 5 specialized agents
- Automatic query routing and complexity analysis
- Query planning and decomposition for complex queries
- Multi-strategy document retrieval with optimization
- Result synthesis and deduplication
- Response validation and quality scoring
- Context preservation across conversations
- Fallback to basic RAG on agent failure
- Performance monitoring under 300ms overhead

Example:
    Using the multi-agent coordinator::

        from agents.coordinator import MultiAgentCoordinator
        from llama_index.core.memory import ChatMemoryBuffer

        coordinator = MultiAgentCoordinator(llm, tools_data)
        response = coordinator.process_query(
            "Compare AI vs ML techniques",
            context=ChatMemoryBuffer.from_defaults()
        )
        print(response.content)
"""

import asyncio
import time
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from llama_index.core.memory import ChatMemoryBuffer
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.tools import (
    plan_query,
    retrieve_documents,
    route_query,
    synthesize_results,
    validate_response,
)


class AgentResponse(BaseModel):
    """Response from multi-agent system."""

    content: str = Field(description="Generated response content")
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="Source documents"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Agent processing metadata"
    )
    validation_score: float = Field(
        default=0.0, description="Response validation confidence"
    )
    processing_time: float = Field(
        default=0.0, description="Total processing time in seconds"
    )


class MultiAgentState(MessagesState):
    """Extended state for multi-agent coordination."""

    # Core state
    tools_data: dict[str, Any] = Field(default_factory=dict)
    context: ChatMemoryBuffer | None = None

    # Agent decisions and results
    routing_decision: dict[str, Any] = Field(default_factory=dict)
    planning_output: dict[str, Any] = Field(default_factory=dict)
    retrieval_results: list[dict[str, Any]] = Field(default_factory=list)
    synthesis_result: dict[str, Any] = Field(default_factory=dict)
    validation_result: dict[str, Any] = Field(default_factory=dict)

    # Performance tracking
    agent_timings: dict[str, float] = Field(default_factory=dict)
    total_start_time: float = Field(default=0.0)

    # Error handling
    errors: list[str] = Field(default_factory=list)
    fallback_used: bool = Field(default=False)


class MultiAgentCoordinator:
    """Main coordinator for multi-agent document analysis system.

    Orchestrates five specialized agents using LangGraph supervisor pattern:
    - Router Agent: Analyzes queries and determines processing strategy
    - Planner Agent: Decomposes complex queries into sub-tasks
    - Retrieval Agent: Executes document retrieval with optimizations
    - Synthesis Agent: Combines and deduplicates multi-source results
    - Validation Agent: Validates response quality and accuracy

    Provides automatic fallback to basic RAG on agent failure and maintains
    conversation context across interactions.
    """

    def __init__(
        self,
        llm: Any,
        tools_data: dict[str, Any],
        enable_fallback: bool = True,
        max_agent_timeout: float = 5.0,
    ):
        """Initialize multi-agent coordinator.

        Args:
            llm: Language model for agent operations
            tools_data: Dictionary containing vector index, KG index, and retriever
            enable_fallback: Whether to fallback to basic RAG on agent failure
            max_agent_timeout: Maximum time to wait for agent responses (seconds)
        """
        self.llm = llm
        self.tools_data = tools_data
        self.enable_fallback = enable_fallback
        self.max_agent_timeout = max_agent_timeout

        # Performance tracking
        self.total_queries = 0
        self.successful_queries = 0
        self.fallback_queries = 0
        self.avg_processing_time = 0.0

        # Create memory for state persistence
        self.memory = InMemorySaver()

        # Initialize agent graph
        self._setup_agent_graph()

        logger.info("MultiAgentCoordinator initialized successfully")

    def _setup_agent_graph(self) -> None:
        """Setup LangGraph supervisor with specialized agents."""
        try:
            # Create individual agents with specific tools
            router_agent = create_react_agent(
                self.llm,
                tools=[route_query],
                state_schema=MultiAgentState,
            )

            planner_agent = create_react_agent(
                self.llm,
                tools=[plan_query],
                state_schema=MultiAgentState,
            )

            retrieval_agent = create_react_agent(
                self.llm,
                tools=[retrieve_documents],
                state_schema=MultiAgentState,
            )

            synthesis_agent = create_react_agent(
                self.llm,
                tools=[synthesize_results],
                state_schema=MultiAgentState,
            )

            validation_agent = create_react_agent(
                self.llm,
                tools=[validate_response],
                state_schema=MultiAgentState,
            )

            # Define agent members for supervisor
            members = [
                "router_agent",
                "planner_agent",
                "retrieval_agent",
                "synthesis_agent",
                "validation_agent",
            ]

            # Create supervisor system prompt
            system_prompt = (
                "You are a supervisor managing a team of specialized document "
                "analysis agents.\n\n"
                "Your team consists of:\n"
                "- router_agent: Analyzes queries and determines processing strategy\n"
                "- planner_agent: Decomposes complex queries into manageable "
                "sub-tasks\n"
                "- retrieval_agent: Executes document retrieval using optimal "
                "strategies\n"
                "- synthesis_agent: Combines and deduplicates results from multiple "
                "sources\n"
                "- validation_agent: Validates response quality and accuracy\n\n"
                "Given a user query, coordinate these agents efficiently:\n\n"
                "1. Always start with router_agent to analyze the query\n"
                "2. If routing indicates needs_planning=true, use planner_agent next\n"
                "3. Use retrieval_agent for document search (may be called "
                "multiple times)\n"
                "4. If multiple retrieval results need combining, use synthesis_agent\n"
                "5. Always end with validation_agent to ensure response quality\n\n"
                "Route efficiently - don't use unnecessary agents. Simple queries "
                "may only need router → retrieval → validation.\n\n"
                "Respond with the agent name to call next, or 'FINISH' when complete."
            )

            # Create supervisor graph
            self.graph = create_supervisor(
                llm=self.llm,
                members=members,
                system_prompt=system_prompt,
            )

            # Add individual agents to the graph
            self.graph.add_node("router_agent", router_agent)
            self.graph.add_node("planner_agent", planner_agent)
            self.graph.add_node("retrieval_agent", retrieval_agent)
            self.graph.add_node("synthesis_agent", synthesis_agent)
            self.graph.add_node("validation_agent", validation_agent)

            # Compile graph with memory
            self.compiled_graph = self.graph.compile(checkpointer=self.memory)

            logger.info("Agent graph setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup agent graph: {e}")
            raise RuntimeError(f"Agent graph initialization failed: {e}") from e

    def process_query(
        self,
        query: str,
        context: ChatMemoryBuffer | None = None,
        settings_override: dict[str, Any] | None = None,
        thread_id: str = "default",
    ) -> AgentResponse:
        """Process user query through multi-agent pipeline.

        Coordinates specialized agents to analyze, plan, retrieve, synthesize,
        and validate responses to user queries. Automatically falls back to
        basic RAG if agent coordination fails.

        Args:
            query: User query to process
            context: Optional conversation context for continuity
            settings_override: Optional settings to override defaults
            thread_id: Thread ID for conversation continuity

        Returns:
            AgentResponse with content, sources, metadata, and validation score

        Example:
            >>> response = coordinator.process_query("What is machine learning?")
            >>> print(response.content)
            >>> print(f"Validation score: {response.validation_score}")
        """
        start_time = time.perf_counter()
        self.total_queries += 1

        try:
            # Initialize state
            initial_state = MultiAgentState(
                messages=[HumanMessage(content=query)],
                tools_data=self.tools_data,
                context=context,
                total_start_time=start_time,
            )

            # Run multi-agent workflow
            result = self._run_agent_workflow(initial_state, thread_id)

            # Extract response from final state
            response = self._extract_response(result, query, start_time)

            # Update performance metrics
            self.successful_queries += 1
            processing_time = time.perf_counter() - start_time
            self._update_performance_metrics(processing_time)

            logger.info(f"Query processed successfully in {processing_time:.3f}s")
            return response

        except Exception as e:
            logger.error(f"Multi-agent processing failed: {e}")

            # Fallback to basic RAG if enabled
            if self.enable_fallback:
                return self._fallback_basic_rag(query, context, start_time)
            else:
                # Return error response
                processing_time = time.perf_counter() - start_time
                return AgentResponse(
                    content=f"Error processing query: {str(e)}",
                    sources=[],
                    metadata={"error": str(e), "fallback_available": False},
                    validation_score=0.0,
                    processing_time=processing_time,
                )

    def _run_agent_workflow(
        self, initial_state: MultiAgentState, thread_id: str
    ) -> dict[str, Any]:
        """Run the multi-agent workflow with timeout protection."""
        try:
            # Create async event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run workflow with timeout
            config = {"configurable": {"thread_id": thread_id}}

            # Execute workflow
            result = None
            for state in self.compiled_graph.stream(
                initial_state,
                config=config,
                stream_mode="values",
            ):
                result = state

                # Check for timeout
                elapsed = time.perf_counter() - initial_state.total_start_time
                if elapsed > self.max_agent_timeout:
                    logger.warning(f"Agent workflow timeout after {elapsed:.2f}s")
                    break

            return result or initial_state

        except Exception as e:
            logger.error(f"Agent workflow execution failed: {e}")
            raise

    def _extract_response(
        self,
        final_state: dict[str, Any],
        original_query: str,
        start_time: float,
    ) -> AgentResponse:
        """Extract and format response from final agent state."""
        try:
            # Get the last message as response content
            messages = final_state.get("messages", [])
            if messages:
                last_message = messages[-1]
                content = getattr(last_message, "content", str(last_message))
            else:
                content = "No response generated by agents"

            # Extract sources from retrieval/synthesis results
            sources = []
            synthesis_result = final_state.get("synthesis_result", {})
            if synthesis_result and "documents" in synthesis_result:
                sources = synthesis_result["documents"]
            elif final_state.get("retrieval_results"):
                # Use last retrieval result
                last_retrieval = final_state["retrieval_results"][-1]
                if "documents" in last_retrieval:
                    sources = last_retrieval["documents"]

            # Get validation score
            validation_result = final_state.get("validation_result", {})
            validation_score = validation_result.get("confidence", 0.0)

            # Build metadata
            processing_time = time.perf_counter() - start_time
            metadata = {
                "routing_decision": final_state.get("routing_decision", {}),
                "planning_output": final_state.get("planning_output", {}),
                "agent_timings": final_state.get("agent_timings", {}),
                "validation_result": validation_result,
                "processing_time_ms": round(processing_time * 1000, 2),
                "agents_used": list(final_state.get("agent_timings", {}).keys()),
                "fallback_used": final_state.get("fallback_used", False),
                "errors": final_state.get("errors", []),
            }

            return AgentResponse(
                content=content,
                sources=sources,
                metadata=metadata,
                validation_score=validation_score,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Failed to extract response: {e}")
            processing_time = time.perf_counter() - start_time
            return AgentResponse(
                content=f"Error extracting response: {str(e)}",
                sources=[],
                metadata={"extraction_error": str(e)},
                validation_score=0.0,
                processing_time=processing_time,
            )

    def _fallback_basic_rag(
        self,
        query: str,
        context: ChatMemoryBuffer | None,
        start_time: float,
    ) -> AgentResponse:
        """Fallback to basic RAG when multi-agent system fails."""
        try:
            self.fallback_queries += 1
            logger.info("Using basic RAG fallback")

            # Use simple vector search as fallback
            from src.agents.tool_factory import ToolFactory

            vector_index = self.tools_data.get("vector")
            if not vector_index:
                raise RuntimeError("No vector index available for fallback")

            # Create basic search tool
            search_tool = ToolFactory.create_vector_search_tool(vector_index)

            # Execute search
            search_result = search_tool.call(query)

            # Format as basic response
            if hasattr(search_result, "response"):
                content = search_result.response
                sources = getattr(search_result, "source_nodes", [])
                # Convert source nodes to dict format
                source_dicts = []
                for node in sources:
                    if hasattr(node, "text") and hasattr(node, "metadata"):
                        source_dicts.append(
                            {
                                "content": node.text,
                                "metadata": node.metadata,
                                "score": getattr(node, "score", 1.0),
                            }
                        )
            else:
                content = str(search_result)
                sources = []
                source_dicts = []

            processing_time = time.perf_counter() - start_time

            return AgentResponse(
                content=content,
                sources=source_dicts,
                metadata={
                    "fallback_used": True,
                    "fallback_strategy": "basic_vector_search",
                    "processing_time_ms": round(processing_time * 1000, 2),
                },
                validation_score=0.7,  # Lower confidence for fallback
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Fallback RAG also failed: {e}")
            processing_time = time.perf_counter() - start_time
            return AgentResponse(
                content=f"System temporarily unavailable. Error: {str(e)}",
                sources=[],
                metadata={
                    "fallback_used": True,
                    "fallback_failed": True,
                    "error": str(e),
                },
                validation_score=0.0,
                processing_time=processing_time,
            )

    def _update_performance_metrics(self, processing_time: float) -> None:
        """Update performance tracking metrics."""
        # Update running average
        if self.total_queries > 0:
            self.avg_processing_time = (
                self.avg_processing_time * (self.total_queries - 1) + processing_time
            ) / self.total_queries
        else:
            self.avg_processing_time = processing_time

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for monitoring.

        Returns:
            Dictionary containing performance metrics including success rates,
            processing times, and fallback usage statistics
        """
        success_rate = (
            self.successful_queries / self.total_queries
            if self.total_queries > 0
            else 0.0
        )
        fallback_rate = (
            self.fallback_queries / self.total_queries
            if self.total_queries > 0
            else 0.0
        )

        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "fallback_queries": self.fallback_queries,
            "success_rate": round(success_rate, 3),
            "fallback_rate": round(fallback_rate, 3),
            "avg_processing_time": round(self.avg_processing_time, 3),
            "agent_timeout": self.max_agent_timeout,
            "fallback_enabled": self.enable_fallback,
        }

    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics."""
        self.total_queries = 0
        self.successful_queries = 0
        self.fallback_queries = 0
        self.avg_processing_time = 0.0
        logger.info("Performance statistics reset")


# Legacy compatibility function
def create_multi_agent_coordinator(
    llm: Any,
    tools_data: dict[str, Any],
    enable_fallback: bool = True,
) -> MultiAgentCoordinator:
    """Create multi-agent coordinator (legacy compatibility).

    Args:
        llm: Language model for agent operations
        tools_data: Dictionary containing retrieval indexes and tools
        enable_fallback: Whether to enable fallback to basic RAG

    Returns:
        Configured MultiAgentCoordinator instance
    """
    return MultiAgentCoordinator(
        llm=llm,
        tools_data=tools_data,
        enable_fallback=enable_fallback,
    )
