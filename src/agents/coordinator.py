"""Multi-Agent Coordination System using LangGraph supervisor.

This module implements a MultiAgentCoordinator that orchestrates five specialized agents
using langgraph-supervisor for query processing with parallel tool execution and
context management.

Features:
- LangGraph supervisor with parallel tool execution
- Structured output mode for consistent response formatting
- Forward message tool for direct agent communication
- Context trimming hooks for memory management
- DSPy integration for query optimization
- Performance tracking and timeout protection

Example:
    Using the multi-agent coordinator:

        from agents.coordinator import MultiAgentCoordinator
        from llama_index.core.memory import ChatMemoryBuffer

        coordinator = MultiAgentCoordinator()
        response = coordinator.process_query(
            "Compare AI vs ML techniques",
            context=ChatMemoryBuffer.from_defaults()
        )
        # response.content contains the generated text
"""

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool

# Import LlamaIndex Settings for context window access
from llama_index.core.memory import ChatMemoryBuffer
from loguru import logger

from src.agents.tools import (
    plan_query,
    route_query,
    router_tool,
    synthesize_results,
    validate_response,
)
from src.config import settings
from src.core.analytics import AnalyticsConfig, AnalyticsManager
from src.dspy_integration import DSPyLlamaIndexRetriever, is_dspy_available

# Import agent-specific models
from .models import AgentResponse, MultiAgentState


class ContextManager:
    """Context manager for 128K context handling and token estimation."""

    def __init__(self, max_context_tokens: int = 131072) -> None:
        """Initialize context manager with 128K context support."""
        self.max_context_tokens = max_context_tokens
        self.trim_threshold = int(max_context_tokens * 0.9)  # 90% threshold
        self.kv_cache_memory_per_token = 1024  # bytes per token for FP8

    def estimate_tokens(self, messages: list[dict] | list[Any]) -> int:
        """Estimate token count for messages (4 chars per token average)."""
        if not messages:
            return 0

        total_chars = 0
        for msg in messages:
            if hasattr(msg, "content"):
                content = str(msg.content)
            elif isinstance(msg, dict) and "content" in msg:
                content = str(msg["content"])
            else:
                content = str(msg)
            total_chars += len(content)

        return total_chars // 4  # 4 chars per token estimate

    def calculate_kv_cache_usage(self, state: dict) -> float:
        """Calculate KV cache memory usage in GB."""
        messages = state.get("messages", [])
        tokens = self.estimate_tokens(messages)
        usage_bytes = tokens * self.kv_cache_memory_per_token
        return usage_bytes / (1024**3)  # Convert to GB

    def structure_response(self, response: Any) -> dict[str, Any]:
        """Structure response with metadata."""
        return {
            "content": str(response),
            "structured": True,
            "generated_at": time.time(),
            "context_optimized": True,
        }


# Constants

COORDINATION_OVERHEAD_THRESHOLD = 0.2  # seconds (200ms target)
CONTEXT_TRIM_STRATEGY = "last"
PARALLEL_TOOL_CALLS_ENABLED = True
OUTPUT_MODE_STRUCTURED = "structured"
CREATE_FORWARD_MESSAGE_TOOL_ENABLED = True
ADD_HANDOFF_BACK_MESSAGES_ENABLED = True


class MultiAgentCoordinator:
    """Coordinator for multi-agent document analysis system.

    Orchestrates five specialized agents using LangGraph supervisor pattern:
    - Router Agent: Analyzes queries and determines processing strategy
    - Planner Agent: Decomposes complex queries into sub-tasks
    - Retrieval Agent: Executes document retrieval with DSPy optimization
    - Synthesis Agent: Combines and deduplicates multi-source results
    - Validation Agent: Validates response quality and accuracy

    Provides context management, performance tracking, and timeout protection.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-4B-Instruct-2507-FP8",
        *,
        max_context_length: int = settings.vllm.context_window,
        backend: str = "vllm",
        enable_fallback: bool = True,
        max_agent_timeout: float = settings.agents.decision_timeout,
    ):
        """Initialize multi-agent coordinator.

        Args:
            model_path: Model path for LLM
            max_context_length: Maximum context in tokens
            backend: Model backend ("vllm" recommended)
            enable_fallback: Whether to fallback to basic RAG on agent failure
            max_agent_timeout: Maximum time for agent responses (seconds)
        """
        self.model_path = model_path
        self.max_context_length = max_context_length
        self.backend = backend
        self.enable_fallback = enable_fallback
        self.max_agent_timeout = max_agent_timeout

        # vLLM Configuration with unified settings
        env_vars = settings.get_vllm_env_vars()

        self.vllm_config = {
            "model": model_path,
            "max_model_len": max_context_length,
            **env_vars,
        }

        # Context management with unified settings
        self.context_window = settings.vllm.context_window
        self.max_tokens = settings.vllm.max_tokens
        self.context_manager = ContextManager(
            max_context_tokens=self.max_context_length
        )

        # Performance tracking (ADR-011)
        self.total_queries = 0
        self.successful_queries = 0
        self.fallback_queries = 0
        self.avg_processing_time = 0.0
        self.avg_coordination_overhead = 0.0

        # Create memory for state persistence
        self.memory = InMemorySaver()

        # Initialize components
        self.llm = None
        self.dspy_retriever = None
        self.compiled_graph = None
        self.graph = None
        self.agents = {}

        # Lazy initialization
        self._setup_complete = False

        logger.info("MultiAgentCoordinator initialized (model: %s)", model_path)

    def _ensure_setup(self) -> bool:
        """Ensure all components are set up (lazy initialization)."""
        if self._setup_complete:
            return True

        try:
            # Initialize vLLM environment for FP8 optimization
            from src.config import setup_llamaindex

            setup_llamaindex()
            logger.info("vLLM configuration applied via unified settings")

            # Use LLM from unified configuration (LlamaIndex Settings)
            from llama_index.core import Settings

            if Settings.llm is not None:
                self.llm = Settings.llm
                logger.info("LLM initialized from LlamaIndex Settings")
            else:
                # Raise error if LLM not properly configured
                raise RuntimeError(
                    "LLM not properly configured in unified settings. "
                    "Please ensure LlamaIndex Settings are initialized."
                )

            # Initialize DSPy integration (ADR-018)
            if is_dspy_available():
                self.dspy_retriever = DSPyLlamaIndexRetriever(llm=self.llm)
                logger.info("Real DSPy integration initialized")
            else:
                logger.warning("DSPy not available - using fallback optimization")

            # Setup agent graph with ADR-011 compliance
            self._setup_agent_graph()

            self._setup_complete = True
            return True

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("Failed to setup coordinator: %s", e)
            return False

    def _setup_agent_graph(self) -> None:
        """Setup LangGraph supervisor with agent orchestration."""
        try:
            # Create individual agents with proper naming and tools
            router_agent = create_react_agent(
                self.llm,
                tools=[route_query],
                state_schema=MultiAgentState,
                name="router_agent",
            )

            planner_agent = create_react_agent(
                self.llm,
                tools=[plan_query],
                state_schema=MultiAgentState,
                name="planner_agent",
            )

            retrieval_agent = create_react_agent(
                self.llm,
                tools=[router_tool],
                state_schema=MultiAgentState,
                name="retrieval_agent",
            )

            synthesis_agent = create_react_agent(
                self.llm,
                tools=[synthesize_results],
                state_schema=MultiAgentState,
                name="synthesis_agent",
            )

            validation_agent = create_react_agent(
                self.llm,
                tools=[validate_response],
                state_schema=MultiAgentState,
                name="validation_agent",
            )

            # Create supervisor system prompt
            system_prompt = self._create_supervisor_prompt()

            # Create list of agents for supervisor
            agents = [
                router_agent,
                planner_agent,
                retrieval_agent,
                synthesis_agent,
                validation_agent,
            ]

            # Create forward message tool for direct communication
            forward_tool = create_forward_message_tool("supervisor")

            # Create supervisor with optimization parameters
            self.graph = create_supervisor(
                agents=agents,
                model=self.llm,
                prompt=system_prompt,
                parallel_tool_calls=PARALLEL_TOOL_CALLS_ENABLED,
                output_mode=OUTPUT_MODE_STRUCTURED,
                create_forward_message_tool=CREATE_FORWARD_MESSAGE_TOOL_ENABLED,
                add_handoff_back_messages=ADD_HANDOFF_BACK_MESSAGES_ENABLED,
                pre_model_hook=self._create_pre_model_hook(),
                post_model_hook=self._create_post_model_hook(),
                tools=[forward_tool],
            )

            # Store agents for reference
            self.agents = {
                "router_agent": router_agent,
                "planner_agent": planner_agent,
                "retrieval_agent": retrieval_agent,
                "synthesis_agent": synthesis_agent,
                "validation_agent": validation_agent,
            }

            # Compile graph with memory
            self.compiled_graph = self.graph.compile(checkpointer=self.memory)

            logger.info("Agent graph setup completed successfully")

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("Failed to setup agent graph: %s", e)
            raise RuntimeError(f"Agent graph initialization failed: {e}") from e

    def _create_supervisor_prompt(self) -> str:
        """Create system prompt for supervisor agent."""
        return (
            "You are a supervisor managing a team of specialized document analysis\n"
            "agents with parallel execution capabilities.\n\n"
            "Team composition:\n"
            "- router_agent: Query analysis and strategy determination\n"
            "- planner_agent: Complex query decomposition\n"
            "- retrieval_agent: Document search with DSPy optimization\n"
            "- synthesis_agent: Multi-source result combination\n"
            "- validation_agent: Response quality validation\n\n"
            "Coordination strategy:\n"
            "1. Always start with router_agent for strategy analysis\n"
            "2. Use planner_agent only if needs_planning=true\n"
            "3. Execute retrieval_agent (may run parallel tool calls)\n"
            "4. Use synthesis_agent for multi-source results\n"
            "5. Always end with validation_agent for quality assurance\n\n"
            "Optimize for parallel execution and minimize unnecessary agent calls.\n\n"
            "Respond with agent name or 'FINISH' when complete."
        )

    def _create_pre_model_hook(self) -> Callable[[dict], dict]:
        """Create pre-model hook for context trimming."""

        def pre_model_hook(state: dict) -> dict:
            """Trim context before model processing."""
            try:
                # Import here so tests patching the function can intercept calls
                from langchain_core.messages.utils import trim_messages

                messages = state.get("messages", [])
                total_tokens = self.context_manager.estimate_tokens(messages)

                if total_tokens > self.context_manager.trim_threshold:
                    # Use intelligent trimming strategy
                    trimmed_messages = trim_messages(
                        messages,
                        strategy=CONTEXT_TRIM_STRATEGY,
                        token_counter=self.context_manager.estimate_tokens,
                        max_tokens=self.context_manager.trim_threshold,
                        start_on="human",
                        end_on=("human", "tool"),
                    )

                    state["messages"] = trimmed_messages
                    state["context_trimmed"] = True
                    state["tokens_trimmed"] = (
                        total_tokens
                        - self.context_manager.estimate_tokens(trimmed_messages)
                    )

                    logger.debug(
                        f"Context trimmed: {state['tokens_trimmed']} tokens removed"
                    )

                return state
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.warning("Pre-model hook failed: %s", e)
                return state

        return pre_model_hook

    def _create_post_model_hook(self) -> Callable[[dict], dict]:
        """Create post-model hook for response formatting."""

        def post_model_hook(state: dict) -> dict:
            """Format response after model generation with structured output."""
            try:
                if state.get("output_mode") == "structured":
                    # Add performance metadata
                    state["optimization_metrics"] = {
                        "context_used_tokens": self.context_manager.estimate_tokens(
                            state.get("messages", [])
                        ),
                        "kv_cache_usage_gb": (
                            self.context_manager.calculate_kv_cache_usage(state)
                        ),
                        "parallel_execution_active": state.get(
                            "parallel_tool_calls", False
                        ),
                        "optimization_enabled": True,
                        "model_path": self.model_path,
                        "context_trimmed": state.get("context_trimmed", False),
                        "tokens_trimmed": state.get("tokens_trimmed", 0),
                    }

                    # Structure response with metadata
                    if "response" in state:
                        state["response"] = self.context_manager.structure_response(
                            state["response"]
                        )

                return state
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.warning("Post-model hook failed: %s", e)
                return state

        return post_model_hook

    def process_query(
        self,
        query: str,
        context: ChatMemoryBuffer | None = None,
        settings_override: dict[str, Any] | None = None,
        thread_id: str = "default",
    ) -> AgentResponse:
        """Process user query through multi-agent pipeline.

        Coordinates specialized agents with parallel tool execution and
        context management.

        Args:
            query: User query to process
            context: Optional conversation context for continuity
            settings_override: Optional settings to override defaults
            thread_id: Thread ID for conversation continuity

        Returns:
            AgentResponse with content, sources, metadata, and performance metrics

        Example:
            >>> response = coordinator.process_query("What is machine learning?")
            >>> response.content  # formatted content string
        """
        start_time = time.perf_counter()
        self.total_queries += 1

        # Ensure setup is complete
        if not self._ensure_setup():
            return self._create_error_response(
                "Failed to initialize coordinator", start_time
            )

        try:
            # Compose tools_data overrides for agent configuration
            defaults: dict[str, Any] = {
                "enable_dspy": settings.enable_dspy_optimization,
                "enable_graphrag": (
                    getattr(settings, "enable_graphrag", False)
                    or settings.get_graphrag_config().get("enabled", False)
                ),
                "enable_multimodal": getattr(settings, "enable_multimodal", False),
                "reranker_normalize_scores": (
                    settings.retrieval.reranker_normalize_scores
                ),
                "reranking_top_k": settings.retrieval.reranking_top_k,
            }

            # Merge caller-provided overrides last (they win)
            tools_data: dict[str, Any] = {**defaults, **(settings_override or {})}

            # Initialize state with execution parameters
            initial_state = MultiAgentState(
                messages=[HumanMessage(content=query)],
                tools_data=tools_data,
                context=context,
                total_start_time=start_time,
                output_mode="structured",
                parallel_execution_active=True,
            )

            # Run multi-agent workflow with performance tracking
            coordination_start = time.perf_counter()
            result = self._run_agent_workflow(initial_state, thread_id)
            coordination_time = time.perf_counter() - coordination_start

            # Extract response from final state
            response = self._extract_response(
                result, query, start_time, coordination_time
            )

            # Update performance metrics
            self.successful_queries += 1
            processing_time = time.perf_counter() - start_time
            self._update_performance_metrics(processing_time, coordination_time)

            # Best-effort analytics logging (never impact user flow)
            if getattr(settings, "analytics_enabled", False):
                with contextlib.suppress(Exception):
                    cfg = AnalyticsConfig(
                        enabled=True,
                        db_path=(
                            settings.analytics_db_path
                            or (settings.data_dir / "analytics" / "analytics.duckdb")
                        ),
                        retention_days=settings.analytics_retention_days,
                    )
                    am = AnalyticsManager.instance(cfg)
                    am.log_query(
                        query_type="chat",
                        latency_ms=processing_time * 1000.0,
                        result_count=0,
                        retrieval_strategy="hybrid",
                        success=True,
                    )

            # Validate performance targets
            if coordination_time > COORDINATION_OVERHEAD_THRESHOLD:
                logger.warning(
                    "Coordination overhead %.3fs exceeds threshold",
                    coordination_time,
                )

            logger.info(
                "Query processed successfully in %.3fs (coordination: %.3fs)",
                processing_time,
                coordination_time,
            )
            return response

        except (RuntimeError, ValueError, AttributeError, TimeoutError) as e:
            logger.error("Multi-agent processing failed: %s", e)

            # Fallback to basic RAG if enabled
            if self.enable_fallback:
                return self._fallback_basic_rag(query, context, start_time)
            return self._create_error_response(str(e), start_time)

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

            # Execute workflow with parallel optimization
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
                    logger.warning("Agent workflow timeout after %.2fs", elapsed)
                    break

            return result or initial_state

        except (RuntimeError, ValueError, AttributeError, TimeoutError) as e:
            logger.error("Agent workflow execution failed: %s", e)
            raise

    def _extract_response(
        self,
        final_state: dict[str, Any],
        _original_query: str,
        start_time: float,
        coordination_time: float,
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

            # Build performance metrics
            processing_time = time.perf_counter() - start_time
            optimization_metrics = {
                "coordination_overhead_ms": round(coordination_time * 1000, 2),
                "meets_target": coordination_time < COORDINATION_OVERHEAD_THRESHOLD,
                "parallel_execution_active": final_state.get(
                    "parallel_execution_active", False
                ),
                "token_reduction_achieved": final_state.get(
                    "token_reduction_achieved", 0.0
                ),
                "context_trimmed": final_state.get("context_trimmed", False),
                "tokens_trimmed": final_state.get("tokens_trimmed", 0),
                "kv_cache_usage_gb": final_state.get("kv_cache_usage_gb", 0.0),
                "model_path": self.model_path,
                "context_window_used": self.max_context_length,
                "optimization_enabled": True,
            }

            # Build metadata
            metadata = {
                "routing_decision": final_state.get("routing_decision", {}),
                "planning_output": final_state.get("planning_output", {}),
                "agent_timings": final_state.get("agent_timings", {}),
                "validation_result": validation_result,
                "processing_time_ms": round(processing_time * 1000, 2),
                "agents_used": list(final_state.get("agent_timings", {}).keys()),
                "fallback_used": final_state.get("fallback_used", False),
                "errors": final_state.get("errors", []),
                "system_info": {
                    "agents_used": 5,
                    "model": self.model_path,
                    "framework": "LangGraph Supervisor",
                    "dspy_available": is_dspy_available(),
                },
            }

            return AgentResponse(
                content=content,
                sources=sources,
                metadata=metadata,
                validation_score=validation_score,
                processing_time=processing_time,
                optimization_metrics=optimization_metrics,
            )

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("Failed to extract response: %s", e)
            processing_time = time.perf_counter() - start_time
            return AgentResponse(
                content=f"Error extracting response: {e!s}",
                sources=[],
                metadata={"extraction_error": str(e)},
                validation_score=0.0,
                processing_time=processing_time,
                optimization_metrics={"error": True},
            )

    def _create_error_response(
        self, error_msg: str, start_time: float
    ) -> AgentResponse:
        """Create error response with timing information."""
        processing_time = time.perf_counter() - start_time
        return AgentResponse(
            content=f"Error processing query: {error_msg}",
            sources=[],
            metadata={"error": error_msg, "fallback_available": self.enable_fallback},
            validation_score=0.0,
            processing_time=processing_time,
            optimization_metrics={"initialization_failed": True},
        )

    def _fallback_basic_rag(
        self,
        query: str,
        _context: ChatMemoryBuffer | None,
        start_time: float,
    ) -> AgentResponse:
        """Fallback to basic RAG when multi-agent system fails."""
        try:
            # Update fallback counter (production behavior)
            self.fallback_queries += 1
            logger.info("Using basic RAG fallback")

            # Simple fallback response
            processing_time = time.perf_counter() - start_time

            return AgentResponse(
                content=(
                    f"I understand you're asking about: {query}. However, the "
                    f"advanced multi-agent system is currently unavailable. Please try "
                    f"again later."
                ),
                sources=[],
                metadata={
                    "fallback_used": True,
                    "fallback_strategy": "basic_response",
                    "processing_time_ms": round(processing_time * 1000, 2),
                },
                validation_score=0.3,  # Lower confidence for fallback
                processing_time=processing_time,
                optimization_metrics={"fallback_mode": True},
            )

        except (RuntimeError, ValueError, AttributeError, Exception) as e:  # pylint: disable=broad-exception-caught
            logger.error("Fallback RAG also failed: %s", e)
            processing_time = time.perf_counter() - start_time
            return AgentResponse(
                content=f"System temporarily unavailable. Error: {e!s}",
                sources=[],
                metadata={
                    "fallback_used": True,
                    "fallback_failed": True,
                    "error": str(e),
                },
                validation_score=0.0,
                processing_time=processing_time,
                optimization_metrics={"system_failure": True},
            )

    def _update_performance_metrics(
        self, processing_time: float, coordination_time: float
    ) -> None:
        """Update performance tracking metrics."""
        # Update running averages
        if self.total_queries > 0:
            self.avg_processing_time = (
                self.avg_processing_time * (self.total_queries - 1) + processing_time
            ) / self.total_queries

            self.avg_coordination_overhead = (
                self.avg_coordination_overhead * (self.total_queries - 1)
                + coordination_time
            ) / self.total_queries
        else:
            self.avg_processing_time = processing_time
            self.avg_coordination_overhead = coordination_time

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dictionary containing performance metrics including success rates,
            processing times, and coordination overhead
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
            # Basic metrics
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "fallback_queries": self.fallback_queries,
            "success_rate": round(success_rate, 3),
            "fallback_rate": round(fallback_rate, 3),
            "avg_processing_time": round(self.avg_processing_time, 3),
            "avg_coordination_overhead_ms": round(
                self.avg_coordination_overhead * 1000, 2
            ),
            "meets_target": self.avg_coordination_overhead
            < COORDINATION_OVERHEAD_THRESHOLD,
            "agent_timeout": self.max_agent_timeout,
            "fallback_enabled": self.enable_fallback,
            "model_config": {
                "model_path": self.model_path,
                "max_context_length": self.max_context_length,
                "backend": self.backend,
            },
        }

    def validate_system_status(self) -> dict[str, bool]:
        """Validate system components and performance."""
        return {
            "graph_setup": self._setup_complete and self.compiled_graph is not None,
            "model_configured": bool(self.model_path),
            "performance_optimization": True,
            "dspy_integration": is_dspy_available(),
            "coordination_performance": self.avg_coordination_overhead
            < COORDINATION_OVERHEAD_THRESHOLD,
            "context_support": self.max_context_length >= 131072,
        }

    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics."""
        self.total_queries = 0
        self.successful_queries = 0
        self.fallback_queries = 0
        self.avg_processing_time = 0.0
        self.avg_coordination_overhead = 0.0
        logger.info("Performance statistics reset")


# Factory function for coordinator
def create_multi_agent_coordinator(
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507-FP8",
    max_context_length: int = settings.vllm.context_window,
    enable_fallback: bool = True,
) -> MultiAgentCoordinator:
    """Create multi-agent coordinator.

    Args:
        model_path: Model path for LLM
        max_context_length: Maximum context in tokens
        enable_fallback: Whether to enable fallback to basic RAG

    Returns:
        Configured MultiAgentCoordinator instance
    """
    return MultiAgentCoordinator(
        model_path=model_path,
        max_context_length=max_context_length,
        enable_fallback=enable_fallback,
    )
