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

        from src.agents.coordinator import MultiAgentCoordinator
        coordinator = MultiAgentCoordinator()
        response = coordinator.process_query(
            "Compare AI vs ML techniques",
            context=None
        )
        # response.content contains the generated text
"""

import asyncio
import contextlib
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool

# Import LlamaIndex Settings for context window access
from llama_index.core.utils import get_tokenizer
from loguru import logger
from opentelemetry import metrics, trace
from opentelemetry.trace import Span

# Registry utilities
# Note: tool imports are performed lazily inside _setup_agent_graph to avoid
# importing heavy dependencies at module import time (improves Streamlit tests
# that stub LlamaIndex modules).
from src.agents.registry import DefaultToolRegistry, RetryLlamaIndexLLM, ToolRegistry
from src.agents.tools.memory import (
    MemoryConsolidationPolicy,
    apply_consolidation_policy,
    consolidate_memory_candidates,
    extract_memory_candidates,
)
from src.config import settings
from src.config.langchain_factory import build_chat_model
from src.dspy_integration import DSPyLlamaIndexRetriever, is_dspy_available
from src.telemetry.opentelemetry import configure_observability

# Import agent-specific models
from .models import AgentResponse, MultiAgentGraphState, MultiAgentState

_COORDINATOR_TRACER = trace.get_tracer("docmind.agents.coordinator")
_COORDINATOR_LATENCY = None
_COORDINATOR_COUNTER = None


class ContextManager:
    """Context manager for 128K context handling and token estimation."""

    def __init__(self, max_context_tokens: int = 131072) -> None:
        """Initialize context manager with 128K context support."""
        self.max_context_tokens = max_context_tokens
        self.trim_threshold = int(max_context_tokens * 0.9)  # 90% threshold
        self.kv_cache_memory_per_token = 1024  # bytes per token for FP8
        self._tokenizer = get_tokenizer()

    def estimate_tokens(self, messages: list[dict] | list[Any]) -> int:
        """Estimate token count for messages using the global tokenizer."""
        if not messages:
            return 0

        total_tokens = 0
        for msg in messages:
            content_attr = getattr(msg, "content", None)
            if content_attr is not None:
                content = str(content_attr)
            elif isinstance(msg, dict) and "content" in msg:
                content = str(msg["content"])
            else:
                content = str(msg)
            total_tokens += len(self._tokenizer(content))

        return total_tokens

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
MEMORY_CONSOLIDATION_MAX_WORKERS = 2
MEMORY_CONSOLIDATION_TIMEOUT_S = 10.0


def _shared_llm_attempts() -> int:
    """Return configured retry attempts for the shared LlamaIndex LLM."""
    retries = int(getattr(settings.agents, "max_retries", 0))
    return max(1, retries + 1)


def _shared_llm_retries() -> int:
    """Return configured retry count for LlamaIndex LLMs that support it."""
    return max(0, int(getattr(settings.agents, "max_retries", 0)))


def _supports_native_llamaindex_retries(llm: Any) -> bool:
    """Return True when the LLM exposes a real `max_retries` field.

    Most LlamaIndex LLM implementations are pydantic models and declare
    `max_retries` explicitly. Avoid treating mocks as supporting this by
    checking for a concrete field map.
    """
    fields = getattr(llm, "model_fields", None)
    if isinstance(fields, dict) and "max_retries" in fields:
        return True
    legacy_fields = getattr(llm, "__fields__", None)
    return isinstance(legacy_fields, dict) and "max_retries" in legacy_fields


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
        tool_registry: ToolRegistry | None = None,
        use_shared_llm_client: bool | None = None,
        checkpointer: Any | None = None,
        store: Any | None = None,
    ):
        """Initialize multi-agent coordinator.

        Args:
            model_path: Model path for LLM
            max_context_length: Maximum context in tokens
            backend: Model backend ("vllm" recommended)
            enable_fallback: Whether to fallback to basic RAG on agent failure
            max_agent_timeout: Maximum time for agent responses (seconds)
            tool_registry: Optional tool registry (primarily for testing)
            use_shared_llm_client: Override shared LlamaIndex LLM wrapper flag.
            checkpointer: Optional LangGraph checkpointer (e.g., SqliteSaver) for
                durable thread persistence. Defaults to in-memory saver.
            store: Optional LangGraph BaseStore for long-term memory. When
                provided, tools can access it via ToolRuntime/store.
        """
        configure_observability(settings)
        self.model_path = model_path
        self.max_context_length = max_context_length
        self.backend = backend
        self.enable_fallback = enable_fallback
        self.max_agent_timeout = max_agent_timeout
        self.use_shared_llm_client = (
            settings.agents.use_shared_llm_client
            if use_shared_llm_client is None
            else use_shared_llm_client
        )

        self.tool_registry: ToolRegistry = tool_registry or DefaultToolRegistry()

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

        # Checkpointer + store (ADR-058). Defaults to in-memory for tests.
        self.checkpointer = checkpointer or InMemorySaver()
        self.store = store

        # Initialize components
        self.llm = None
        self.llamaindex_llm = None
        self._shared_llm_wrapper: RetryLlamaIndexLLM | None = None
        self.dspy_retriever = None
        self.compiled_graph = None
        self.graph = None
        self.agents = {}

        # Lazy initialization
        self._setup_complete = False
        self._memory_executor = ThreadPoolExecutor(
            max_workers=MEMORY_CONSOLIDATION_MAX_WORKERS,
            thread_name_prefix="docmind-memory",
        )

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
                self.llamaindex_llm = Settings.llm
                if self.use_shared_llm_client:
                    retries = _shared_llm_retries()
                    # Prefer native retry configuration when supported (e.g.,
                    # OpenAI-like backends expose `max_retries`).
                    if _supports_native_llamaindex_retries(self.llamaindex_llm):
                        try:
                            self.llamaindex_llm.max_retries = retries  # type: ignore[attr-defined]
                            logger.info(
                                "Configured shared LlamaIndex LLM retries: %d",
                                retries,
                            )
                        except (AttributeError, TypeError, ValueError):
                            self._shared_llm_wrapper = RetryLlamaIndexLLM(
                                self.llamaindex_llm,
                                max_attempts=_shared_llm_attempts(),
                            )
                            self.llamaindex_llm = self._shared_llm_wrapper
                    else:
                        self._shared_llm_wrapper = RetryLlamaIndexLLM(
                            self.llamaindex_llm,
                            max_attempts=_shared_llm_attempts(),
                        )
                        self.llamaindex_llm = self._shared_llm_wrapper
                logger.info("LlamaIndex LLM initialized from Settings")
            else:
                # Raise error if LLM not properly configured
                raise RuntimeError(
                    "LLM not properly configured in unified settings. "
                    "Please ensure LlamaIndex Settings are initialized."
                )

            # LangGraph requires a LangChain-compatible model runnable; keep it
            # separate from the LlamaIndex LLM used by DSPy.
            base_chat_model = build_chat_model(settings)
            # ChatOpenAI already supports built-in retries via `max_retries`.
            self.llm = base_chat_model
            logger.info("LangChain chat model initialized from unified settings")

            # Initialize DSPy integration (ADR-018)
            if is_dspy_available():
                self.dspy_retriever = DSPyLlamaIndexRetriever(llm=self.llamaindex_llm)
                logger.info("Real DSPy integration initialized")
            else:
                logger.warning("DSPy not available - using fallback optimization")

            # Setup agent graph with ADR-011 compliance
            self._setup_agent_graph()

            self._setup_complete = True
            return True

        except (RuntimeError, ValueError, AttributeError, ImportError) as e:
            logger.error("Failed to setup coordinator ({}): {}", type(e).__name__, e)
            return False

    def _setup_agent_graph(self) -> None:
        """Setup LangGraph supervisor with agent orchestration."""
        try:
            if self.llm is None:
                raise RuntimeError("LangChain chat model is not initialized")
            model = self.llm

            agent_specs: tuple[tuple[str, Callable[[], list[Any]]], ...] = (
                ("router_agent", lambda: list(self.tool_registry.get_router_tools())),
                ("planner_agent", lambda: list(self.tool_registry.get_planner_tools())),
                (
                    "retrieval_agent",
                    lambda: list(self.tool_registry.get_retrieval_tools()),
                ),
                (
                    "synthesis_agent",
                    lambda: list(self.tool_registry.get_synthesis_tools()),
                ),
                (
                    "validation_agent",
                    lambda: list(self.tool_registry.get_validation_tools()),
                ),
            )

            agents: dict[str, Any] = {
                name: create_agent(
                    model,
                    tools=tool_loader(),
                    state_schema=MultiAgentGraphState,
                    name=name,
                    store=self.store,
                )
                for name, tool_loader in agent_specs
            }

            # Create supervisor system prompt
            system_prompt = self._create_supervisor_prompt()

            # Create list of agents for supervisor preserving definition order
            supervisor_agents = [agents[name] for name, _ in agent_specs]

            # Create forward message tool for direct communication
            forward_tool = create_forward_message_tool("supervisor")

            # Create supervisor with corrected flags per ADR-011
            # - output_mode: "last_message" (structured metadata stays in state)
            # - add_handoff_messages: True (handoff propagation)
            # - include forward message tool in tools list
            with warnings.catch_warnings():
                # langgraph-supervisor currently relies on the deprecated
                # `langgraph.prebuilt.create_react_agent`; suppress that warning
                # until an upstream release removes the call (latest as of 0.0.31).
                warnings.filterwarnings(
                    "ignore",
                    message=r"create_react_agent has been moved.*",
                    category=DeprecationWarning,
                )
                self.graph = create_supervisor(
                    agents=supervisor_agents,
                    model=model,
                    prompt=system_prompt,
                    parallel_tool_calls=PARALLEL_TOOL_CALLS_ENABLED,
                    output_mode="last_message",
                    add_handoff_messages=True,
                    pre_model_hook=self._create_pre_model_hook(),
                    post_model_hook=self._create_post_model_hook(),
                    tools=[forward_tool],
                )

            # Store agents for reference
            self.agents = agents

            # Compile graph with checkpointer + optional store (ADR-058).
            self.compiled_graph = self.graph.compile(
                checkpointer=self.checkpointer, store=self.store
            )

            logger.info("Agent graph setup completed successfully")

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("Failed to setup agent graph: %s", e)
            raise RuntimeError(f"Agent graph initialization failed: {e}") from e

    def _create_supervisor_prompt(self) -> str:
        """Create system prompt for supervisor agent."""
        return (
            "You are a supervisor managing a team of specialized document analysis\n"
            "agents with parallel execution capabilities.\n\n"
            "Performance target: keep coordination overhead under 200ms per turn.\n\n"
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
                # Defer import to allow runtime patching and reduce import-time deps
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
                # Non-fatal: annotate state for observability only if dict
                if isinstance(state, dict):
                    state["hook_error"] = True
                    state["hook_name"] = "pre_model_hook"
                return state

        return pre_model_hook

    def _create_post_model_hook(self) -> Callable[[dict], dict]:
        """Create post-model hook to attach optimization metrics consistently."""

        def post_model_hook(state: dict) -> dict:
            try:
                # Build base metrics from current state via shared helper
                state["optimization_metrics"] = (
                    self._build_base_optimization_metrics_from_state(state)
                )
                return state
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.warning("Post-model hook failed: %s", e)
                if isinstance(state, dict):
                    state["hook_error"] = True
                    state["hook_name"] = "post_model_hook"
                return state

        return post_model_hook

    def _build_base_optimization_metrics_from_state(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Build base optimization metrics derived only from current state.

        Used by both the post-model hook (per-turn annotation) and the final
        AgentResponse metrics builder to avoid duplication and ensure
        consistency across code paths.
        """
        return {
            "context_used_tokens": self.context_manager.estimate_tokens(
                state.get("messages", [])
            ),
            "kv_cache_usage_gb": self.context_manager.calculate_kv_cache_usage(state),
            # Normalize flag naming across code paths
            "parallel_execution_active": bool(
                state.get("parallel_execution_active")
                or state.get("parallel_tool_calls")
            ),
            "optimization_enabled": True,
            "model_path": self.model_path,
            "context_trimmed": state.get("context_trimmed", False),
            "tokens_trimmed": state.get("tokens_trimmed", 0),
        }

    def _build_optimization_metrics(
        self, final_state: dict[str, Any], coordination_time: float
    ) -> dict[str, Any]:
        """Build optimization metrics for AgentResponse.

        Args:
            final_state: The final agent state containing messages and flags.
            coordination_time: Time spent coordinating agents in seconds.

        Returns:
            A dictionary with optimization metrics fields suitable for
            AgentResponse.optimization_metrics.
        """
        base = self._build_base_optimization_metrics_from_state(final_state)
        base.update(
            {
                "coordination_overhead_ms": round(coordination_time * 1000, 2),
                "meets_target": coordination_time < COORDINATION_OVERHEAD_THRESHOLD,
                "token_reduction_achieved": final_state.get(
                    "token_reduction_achieved", 0.0
                ),
                "context_window_used": self.max_context_length,
            }
        )
        return base

    def _record_query_metrics(self, latency_s: float, success: bool) -> None:
        """Record coordinator latency metrics via OpenTelemetry when available."""
        with contextlib.suppress(Exception):
            global _COORDINATOR_LATENCY
            global _COORDINATOR_COUNTER
            meter = metrics.get_meter(__name__)
            if _COORDINATOR_LATENCY is None:
                _COORDINATOR_LATENCY = meter.create_histogram(
                    "docmind.coordinator.latency",
                    description="Coordinator end-to-end latency",
                    unit="s",
                )
            if _COORDINATOR_COUNTER is None:
                _COORDINATOR_COUNTER = meter.create_counter(
                    "docmind.coordinator.calls",
                    description="Coordinator invocation count",
                )
            attributes = {"success": "true" if success else "false"}
            _COORDINATOR_LATENCY.record(float(latency_s), attributes=attributes)
            _COORDINATOR_COUNTER.add(1, attributes=attributes)

    def _start_span(
        self, exit_stack: contextlib.ExitStack, thread_id: str, query_length: int
    ) -> Span:
        """Start an OpenTelemetry span for query processing with attributes.

        Args:
            exit_stack: Context manager stack for span lifecycle.
            thread_id: Unique thread identifier for tracing.
            query_length: Character length of the input query.

        Returns:
            Active span for the coordinator operation.
        """
        span = exit_stack.enter_context(
            _COORDINATOR_TRACER.start_as_current_span("coordinator.process_query")
        )
        span.set_attribute("coordinator.thread_id", thread_id)
        span.set_attribute("query.length", query_length)
        return span

    def _build_initial_state(
        self,
        query: str,
        start_time: float,
        tools_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the initial state dict for the agent workflow.

        Args:
            query: User query string.
            start_time: Process start timestamp for timing.
            tools_data: Tool configuration data for agents.

        Returns:
            Initial MultiAgentState as a dict.
        """
        return MultiAgentState(
            messages=[HumanMessage(content=query)],
            tools_data=tools_data,
            total_start_time=start_time,
            output_mode="last_message",
            parallel_execution_active=True,
        ).model_dump()

    def _handle_timeout_response(
        self, query: str, context: Any | None, start_time: float
    ) -> tuple[AgentResponse, bool]:
        """Handle workflow timeout by falling back to basic RAG or error response.

        Args:
            query: Original user query.
            context: Optional context (unused in fallback).
            start_time: Query processing start time.

        Returns:
            Tuple of (AgentResponse, used_fallback_flag).
        """
        if self.enable_fallback:
            response = self._fallback_basic_rag(query, context, start_time)
            try:
                response.metadata["reason"] = "timeout"
                response.metadata["fallback_source"] = "basic_rag"
                if isinstance(response.optimization_metrics, dict):
                    response.optimization_metrics["timeout"] = True
                else:
                    response.optimization_metrics = {"timeout": True}
            except (KeyError, TypeError, AttributeError) as exc:
                logger.debug("Failed to annotate fallback metadata: %s", exc)
            return response, True
        processing_time = time.perf_counter() - start_time
        response = AgentResponse(
            content=("The multi-agent system timed out while processing your request."),
            sources=[],
            metadata={"fallback_used": False, "reason": "timeout"},
            validation_score=0.0,
            processing_time=processing_time,
            optimization_metrics={"timeout": True},
        )
        return response, False

    def _handle_workflow_result(
        self,
        result: dict[str, Any] | None,
        query: str,
        context: Any | None,
        start_time: float,
        coordination_time: float,
    ) -> tuple[AgentResponse, bool, bool]:
        """Process the final workflow result or handle timeout fallback.

        Args:
            result: Workflow output state dict, or None on timeout.
            query: Original user query.
            context: Optional context for fallback.
            start_time: Query processing start time.
            coordination_time: Time spent coordinating agents.

        Returns:
            Tuple of (AgentResponse, workflow_timed_out, used_fallback).
        """
        workflow_timed_out = bool(isinstance(result, dict) and result.get("timed_out"))
        if not workflow_timed_out:
            response = self._extract_response(
                result or {}, query, start_time, coordination_time
            )
            return response, False, False

        logger.warning("Coordinator detected timeout; invoking fallback policy")
        response, used_fallback = self._handle_timeout_response(
            query, context, start_time
        )
        self._record_query_metrics(time.perf_counter() - start_time, False)
        return response, True, used_fallback

    def _update_metrics_after_response(
        self,
        workflow_timed_out: bool,
        used_fallback: bool,
        processing_time: float,
        coordination_time: float,
    ) -> None:
        """Update internal performance metrics after processing a response.

        Args:
            workflow_timed_out: Whether the workflow exceeded timeout.
            used_fallback: Whether fallback was used.
            processing_time: Total query processing time.
            coordination_time: Time spent coordinating agents.
        """
        if not workflow_timed_out:
            self.successful_queries += 1
        self._update_performance_metrics(processing_time, coordination_time)

    def _annotate_span(
        self,
        span: Span,
        workflow_timed_out: bool,
        used_fallback: bool,
        processing_time: float,
    ) -> None:
        """Annotate the OpenTelemetry span with query processing outcomes.

        Args:
            span: Active span to annotate.
            workflow_timed_out: Whether the workflow timed out.
            used_fallback: Whether fallback was invoked.
            processing_time: Total processing time in seconds.
        """
        span.set_attribute("coordinator.workflow_timeout", bool(workflow_timed_out))
        span.set_attribute(
            "coordinator.fallback", bool(workflow_timed_out and used_fallback)
        )
        span.set_attribute("coordinator.success", not workflow_timed_out)
        span.set_attribute(
            "coordinator.processing_time_ms", round(processing_time * 1000.0, 3)
        )

    def process_query(
        self,
        query: str,
        context: Any | None = None,
        settings_override: dict[str, Any] | None = None,
        thread_id: str = "default",
        user_id: str = "local",
        checkpoint_id: str | None = None,
    ) -> AgentResponse:
        """Process user query through multi-agent pipeline.

        Coordinates specialized agents with parallel tool execution and
        context management.

        Args:
            query: User query to process
            context: Optional caller-provided context (not persisted).
            settings_override: Optional settings to override defaults
            thread_id: Thread ID for conversation continuity
            user_id: User namespace identifier (used for memory scoping).
            checkpoint_id: Optional checkpoint id to resume from (time travel).

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

        with contextlib.ExitStack() as exit_stack:
            span = self._start_span(exit_stack, thread_id, len(query))

            try:
                tools_data: dict[str, Any] = self.tool_registry.build_tools_data(
                    settings_override
                )

                # Initialize state with execution parameters
                initial_state = self._build_initial_state(query, start_time, tools_data)

                # Run multi-agent workflow with performance tracking
                coordination_start = time.perf_counter()
                result = self._run_agent_workflow(
                    initial_state,
                    thread_id=thread_id,
                    user_id=user_id,
                    checkpoint_id=checkpoint_id,
                    runtime_context=settings_override,
                )
                coordination_time = time.perf_counter() - coordination_start

                response, workflow_timed_out, used_fallback = (
                    self._handle_workflow_result(
                        result, query, context, start_time, coordination_time
                    )
                )

                # Update performance metrics
                processing_time = time.perf_counter() - start_time
                self._update_metrics_after_response(
                    workflow_timed_out,
                    used_fallback,
                    processing_time,
                    coordination_time,
                )

                # Memory consolidation (SPEC-041)
                if not workflow_timed_out:
                    self._schedule_memory_consolidation(
                        result or {}, thread_id=thread_id, user_id=user_id
                    )

                # Best-effort analytics logging (never impact user flow)
                if not workflow_timed_out:
                    self._record_query_metrics(processing_time, True)

                # Validate performance targets
                if coordination_time > COORDINATION_OVERHEAD_THRESHOLD:
                    logger.warning(
                        "Coordination overhead {overhead:.3f}s exceeds threshold",
                        overhead=coordination_time,
                    )

                self._annotate_span(
                    span, workflow_timed_out, used_fallback, processing_time
                )

                logger.info(
                    "Query processed in {processing:.3f}s "
                    "(coordination: {coordination:.3f}s)",
                    processing=processing_time,
                    coordination=coordination_time,
                )
                return response

            except (RuntimeError, ValueError, AttributeError, TimeoutError) as exc:
                span.set_attribute("coordinator.success", False)
                span.set_attribute("coordinator.error", str(exc))
                logger.error("Multi-agent processing failed: {}", exc)

                # Fallback to basic RAG if enabled
                if self.enable_fallback:
                    return self._fallback_basic_rag(query, context, start_time)
                return self._create_error_response(str(exc), start_time)

    def _run_agent_workflow(
        self,
        initial_state: dict[str, Any],
        *,
        thread_id: str,
        user_id: str,
        checkpoint_id: str | None,
        runtime_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Run the multi-agent workflow with timeout protection.

        Marks timeout directly on the returned state (timed_out=True) when
        wall-clock exceeds the configured decision timeout.
        """
        try:
            if self.compiled_graph is None:
                raise RuntimeError("Agent graph is not compiled")

            # Ensure an event loop exists
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            cfg: dict[str, Any] = {"thread_id": str(thread_id), "user_id": str(user_id)}
            if checkpoint_id:
                cfg["checkpoint_id"] = str(checkpoint_id)
            config: RunnableConfig = {"configurable": cfg}
            result: dict[str, Any] | None = None
            graph = cast(Any, self.compiled_graph)
            for state in graph.stream(
                initial_state,
                config=config,
                context=runtime_context,
                stream_mode="values",
            ):
                if isinstance(state, dict):
                    result = cast(dict[str, Any], state)
                else:
                    model_dump = getattr(state, "model_dump", None)
                    if not callable(model_dump):
                        logger.debug(
                            "Unexpected state type %s without model_dump; "
                            "using empty dict",
                            type(state).__name__,
                        )
                    result = cast(
                        dict[str, Any], model_dump() if callable(model_dump) else {}
                    )
                elapsed = time.perf_counter() - float(
                    initial_state.get("total_start_time", 0.0)
                )
                if elapsed > self.max_agent_timeout:
                    logger.warning("Agent workflow timeout after %.2fs", elapsed)
                    try:
                        if isinstance(result, dict):
                            result["timed_out"] = True
                            result["deadline_s"] = float(self.max_agent_timeout)
                    except (TypeError, AttributeError) as exc:
                        logger.debug("Failed to mark timeout flag on state: %s", exc)
                    break

            if result is not None:
                return result
            logger.warning(
                "Agent workflow produced no result; returning initial state "
                "(thread_id=%s)",
                thread_id,
            )
            return initial_state

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
            optimization_metrics = self._build_optimization_metrics(
                final_state, coordination_time
            )

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
        _context: Any | None,
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

        except (RuntimeError, ValueError, AttributeError, Exception) as e:
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
        """Update running averages for processing and coordination times.

        Args:
            processing_time: Total query processing time in seconds.
            coordination_time: Agent coordination overhead in seconds.
        """
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

        stats = {
            # Basic metrics
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "fallback_queries": self.fallback_queries,
            "success_rate": round(success_rate, 3),
            "fallback_rate": round(fallback_rate, 3),
            "avg_processing_time": round(self.avg_processing_time, 3),
            "avg_coordination_overhead": round(self.avg_coordination_overhead, 3),
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
        # Include ADR compliance snapshot for observability
        stats["adr_compliance"] = self.validate_adr_compliance()
        return stats

    # --- Persistence helpers (ADR-058) ---

    def get_state_values(
        self,
        *,
        thread_id: str,
        user_id: str = "local",
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        """Return the persisted state values for a thread (latest by default)."""
        if not self._ensure_setup():
            return {}
        if self.compiled_graph is None:
            return {}
        cfg: dict[str, Any] = {"thread_id": str(thread_id), "user_id": str(user_id)}
        if checkpoint_id:
            cfg["checkpoint_id"] = str(checkpoint_id)
        snap = self.compiled_graph.get_state({"configurable": cfg})
        values = getattr(snap, "values", None)
        return values if isinstance(values, dict) else {}

    def list_checkpoints(
        self, *, thread_id: str, user_id: str = "local", limit: int = 20
    ) -> list[dict[str, Any]]:
        """List recent checkpoints for a thread (newest first)."""
        if not self._ensure_setup():
            return []
        if self.compiled_graph is None:
            return []
        cfg: dict[str, Any] = {"thread_id": str(thread_id), "user_id": str(user_id)}
        out: list[dict[str, Any]] = []
        try:
            for snap in self.compiled_graph.get_state_history(
                {"configurable": cfg}, limit=int(limit)
            ):
                config = getattr(snap, "config", None)
                conf = config.get("configurable") if isinstance(config, dict) else None
                conf = conf if isinstance(conf, dict) else {}
                out.append(
                    {
                        "checkpoint_id": conf.get("checkpoint_id"),
                        "checkpoint_ns": conf.get("checkpoint_ns", ""),
                    }
                )
        except Exception as exc:
            logger.debug("Checkpoint listing failed: %s", exc)
        return out

    def fork_from_checkpoint(
        self,
        *,
        thread_id: str,
        user_id: str = "local",
        checkpoint_id: str,
    ) -> str | None:
        """Fork a new branch head from a prior checkpoint (SPEC-041 / ADR-058)."""
        if not self._ensure_setup():
            return None
        if self.compiled_graph is None:
            return None

        cfg: dict[str, Any] = {
            "thread_id": str(thread_id),
            "user_id": str(user_id),
            "checkpoint_id": str(checkpoint_id),
        }
        config: RunnableConfig = {"configurable": cfg}
        try:
            # LangGraph special update that clones the selected checkpoint into a
            # new head (time-travel fork) without running the workflow.
            new_config = self.compiled_graph.update_state(
                config, None, as_node="__copy__"
            )
        except Exception as exc:
            logger.debug(
                "Checkpoint fork failed (thread_id=%s checkpoint_id=%s): %s",
                thread_id,
                checkpoint_id,
                exc,
            )
            return None

        conf = new_config.get("configurable") if isinstance(new_config, dict) else None
        conf = conf if isinstance(conf, dict) else {}
        new_checkpoint_id = conf.get("checkpoint_id")
        return str(new_checkpoint_id) if new_checkpoint_id else None

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

    def validate_adr_compliance(self) -> dict[str, bool]:
        """Validate a subset of ADR requirements at runtime for visibility.

        Returns a small dictionary with pass/fail flags used by tests and the UI.
        """
        return {
            # ADR-001: Supervisor pattern compiled and ready
            "adr_001_supervisor_pattern": bool(
                self._setup_complete and self.compiled_graph is not None
            ),
            # ADR-004: FP8 model variant used by default (suffix-based)
            "adr_004_fp8_model": self.model_path.upper()
            .split("/")[-1]
            .endswith("-FP8"),
            # Coordination under 200ms per turn
            "coordination_under_200ms": self.avg_coordination_overhead
            < COORDINATION_OVERHEAD_THRESHOLD,
            # Context support for 128k tokens
            "context_128k_support": self.max_context_length >= 131072,
        }

    def _consolidate_memories(
        self,
        final_state: dict[str, Any],
        thread_id: str,
        user_id: str,
    ) -> None:
        """Perform background memory consolidation."""
        logger.debug(
            "Starting background memory consolidation for thread {}", thread_id
        )
        if self.store is None:
            return

        try:
            start_time = time.monotonic()

            def _timed_out() -> bool:
                return (time.monotonic() - start_time) > MEMORY_CONSOLIDATION_TIMEOUT_S

            if _timed_out():
                logger.debug(
                    "Memory consolidation timed out before start for thread {}",
                    thread_id,
                )
                return

            # 1. Get current checkpoint info for source tracking
            config: RunnableConfig = {
                "configurable": {"thread_id": thread_id, "user_id": user_id}
            }
            checkpoint_id = "latest"
            if self.compiled_graph:
                state = self.compiled_graph.get_state(config)
                if state and state.config:
                    checkpoint_id = state.config.get("configurable", {}).get(
                        "checkpoint_id", "latest"
                    )

            if _timed_out():
                logger.debug("Memory consolidation timed out for thread {}", thread_id)
                return

            policy = MemoryConsolidationPolicy(
                similarity_threshold=float(settings.chat.memory_similarity_threshold),
                low_importance_threshold=float(
                    settings.chat.memory_low_importance_threshold
                ),
                low_importance_ttl_minutes=int(
                    settings.chat.memory_low_importance_ttl_days
                )
                * 24
                * 60,
                max_items_per_namespace=int(
                    settings.chat.memory_max_items_per_namespace
                ),
                max_candidates_per_turn=int(
                    settings.chat.memory_max_candidates_per_turn
                ),
            )

            # 2. Extract candidates from the conversation turn
            messages = final_state.get("messages", [])
            candidates = extract_memory_candidates(
                messages, checkpoint_id=checkpoint_id, llm=self.llm, policy=policy
            )
            if not candidates:
                return
            if _timed_out():
                logger.debug("Memory consolidation timed out for thread {}", thread_id)
                return

            # 3. Consolidate within the specific namespace
            # Namespace: ("memories", "{user_id}", "{thread_id}")
            namespace = ("memories", str(user_id), str(thread_id))
            actions = consolidate_memory_candidates(
                candidates, self.store, namespace, policy=policy
            )
            if _timed_out():
                logger.debug("Memory consolidation timed out for thread {}", thread_id)
                return

            # 4. Apply policy using the durable store
            apply_consolidation_policy(self.store, namespace, actions, policy=policy)

        except Exception as exc:
            logger.debug("Memory consolidation background task failed: {}", exc)

    def _schedule_memory_consolidation(
        self,
        final_state: dict[str, Any],
        *,
        thread_id: str,
        user_id: str,
    ) -> None:
        """Schedule memory consolidation off the critical path."""
        if self.store is None:
            return
        self._memory_executor.submit(
            self._consolidate_memories, final_state, thread_id, user_id
        )


# Factory function for coordinator
def create_multi_agent_coordinator(
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507-FP8",
    max_context_length: int = settings.vllm.context_window,
    enable_fallback: bool = True,
    *,
    tool_registry: ToolRegistry | None = None,
    use_shared_llm_client: bool | None = None,
) -> MultiAgentCoordinator:
    """Create multi-agent coordinator.

    Args:
        model_path: Model path for LLM
        max_context_length: Maximum context in tokens
        enable_fallback: Whether to enable fallback to basic RAG
        tool_registry: Optional registry override for tool resolution
        use_shared_llm_client: Override shared LlamaIndex LLM wrapper flag.

    Returns:
        Configured MultiAgentCoordinator instance
    """
    return MultiAgentCoordinator(
        model_path=model_path,
        max_context_length=max_context_length,
        enable_fallback=enable_fallback,
        tool_registry=tool_registry,
        use_shared_llm_client=use_shared_llm_client,
    )
