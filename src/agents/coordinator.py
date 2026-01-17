"""Multi-Agent Coordination System using a graph-native LangGraph supervisor.

This module implements a MultiAgentCoordinator that orchestrates five specialized agents
using a graph-native LangGraph `StateGraph` supervisor for query processing, with
deadline propagation, runtime context injection, and checkpoint/store support.

Features:
- LangGraph `StateGraph` supervisor orchestration
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

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

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

# Graph-native supervisor implementation (repo-local).
from src.agents.supervisor_graph import (
    SupervisorBuildParams,
    build_multi_agent_supervisor_graph,
    create_forward_message_tool,
)
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
from src.utils.log_safety import build_pii_log_entry
from src.utils.telemetry import log_jsonl

# Import agent-specific models
from .models import AgentResponse, MultiAgentGraphState, MultiAgentState

if TYPE_CHECKING:  # pragma: no cover
    from src.utils.semantic_cache import CacheKey

_COORDINATOR_TRACER = trace.get_tracer("docmind.agents.coordinator")
_COORDINATOR_LATENCY = None
_COORDINATOR_COUNTER = None


class ContextManager:
    """Manages context window constraints and token estimations.

    Provides utilities for calculating token counts and estimating KV cache
    memory usage for optimized context handling.
    """

    def __init__(self, max_context_tokens: int = 131072) -> None:
        """Initializes the context manager with specified token limits.

        Args:
            max_context_tokens: Maximum allowed tokens in the context window.
        """
        self.max_context_tokens = max_context_tokens
        self.trim_threshold = int(max_context_tokens * 0.9)  # 90% threshold
        self.kv_cache_memory_per_token = 1024  # bytes per token for FP8
        self._tokenizer = get_tokenizer()

    def estimate_tokens(self, messages: list[dict] | list[Any]) -> int:
        """Estimates the total token count for a sequence of messages.

        Args:
            messages: A list of message objects or dictionaries to analyze.

        Returns:
            The total estimated number of tokens across all messages.
        """
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
        """Calculates the estimated KV cache memory consumption in gigabytes.

        Args:
            state: The current graph state containing message history.

        Returns:
            The estimated memory usage in GB.
        """
        messages = state.get("messages", [])
        tokens = self.estimate_tokens(messages)
        usage_bytes = tokens * self.kv_cache_memory_per_token
        return usage_bytes / (1024**3)  # Convert to GB

    def structure_response(self, response: Any) -> dict[str, Any]:
        """Wraps a raw response with metadata and optimization flags.

        Args:
            response: The raw response data to structure.

        Returns:
            A dictionary containing the response content and associated metadata.
        """
        return {
            "content": str(response),
            "structured": True,
            "generated_at": time.time(),
            "context_optimized": True,
        }


# Constants
COORDINATION_OVERHEAD_THRESHOLD = 0.2  # seconds (200ms target)
CONTEXT_TRIM_STRATEGY = "last"
MEMORY_CONSOLIDATION_MAX_WORKERS = 2
MEMORY_CONSOLIDATION_TIMEOUT_S = 10.0
MEMORY_CONSOLIDATION_RELEASE_GRACE_S = 30.0


class _AgentHookMiddleware(AgentMiddleware[MultiAgentGraphState, Any]):
    """Applies coordinator pre- and post-model hooks as LangChain middleware.

    Integrates context trimming and metrics collection into the agent execution
    flow.
    """

    def __init__(
        self,
        *,
        pre_model_hook: Callable[[dict], dict],
        post_model_hook: Callable[[dict], dict],
    ) -> None:
        """Initializes middleware with transformation hooks.

        Args:
            pre_model_hook: Function to process state before model invocation.
            post_model_hook: Function to process state after model invocation.
        """
        self._pre = pre_model_hook
        self._post = post_model_hook

    def before_model(
        self, state: MultiAgentGraphState, runtime: Any
    ) -> dict[str, Any] | None:
        """Executes the pre-model hook, initiating context trimming if required.

        Args:
            state: The current graph state.
            runtime: The agent runtime environment.

        Returns:
            A state update dictionary containing trimmed messages, or None.
        """
        del runtime
        processed = self._pre(dict(state))
        if not isinstance(processed, dict) or not processed.get("context_trimmed"):
            return None
        trimmed_messages = processed.get("messages")
        if not isinstance(trimmed_messages, list):
            return None
        tokens_trimmed = processed.get("tokens_trimmed", 0)
        return {
            # Replace message history (not append) using REMOVE_ALL_MESSAGES.
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed_messages],
            "context_trimmed": True,
            "tokens_trimmed": tokens_trimmed,
        }

    def after_model(
        self, state: MultiAgentGraphState, runtime: Any
    ) -> dict[str, Any] | None:
        """Executes the post-model hook to collect optimization metrics.

        Args:
            state: The current graph state.
            runtime: The agent runtime environment.

        Returns:
            A state update dictionary containing collected metrics, or None.
        """
        del runtime
        processed = self._post(dict(state))
        if not isinstance(processed, dict):
            return None
        metrics = processed.get("optimization_metrics")
        if not isinstance(metrics, dict):
            return None
        return {"optimization_metrics": metrics}


def _ensure_event_loop() -> None:
    """Ensures a valid asyncio event loop is active in the current thread.

    Provides a fail-safe mechanism to set a new loop if a policy-managed loop
    is unavailable, avoiding deprecated patterns.
    """
    try:
        asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def _as_state_dict(state: Any) -> dict[str, Any]:
    """Normalizes graph state objects into plain dictionaries.

    Args:
        state: The graph state object (e.g., dict or Pydantic model).

    Returns:
        A dictionary representation of the state.
    """
    if isinstance(state, dict):
        return state
    model_dump = getattr(state, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    logger.debug(
        "Unexpected state type {} without model_dump; using empty dict",
        type(state).__name__,
    )
    return {}


def _coerce_deadline_ts(initial_state: dict[str, Any]) -> float | None:
    """Extracts and validates a monotonic deadline timestamp from state.

    Args:
        initial_state: The initial state dictionary containing potential deadlines.

    Returns:
        The deadline as a float timestamp, or None if invalid or missing.
    """
    raw_deadline = initial_state.get("deadline_ts")
    try:
        return float(raw_deadline) if raw_deadline is not None else None
    except (TypeError, ValueError):
        return None


def _shared_llm_attempts() -> int:
    """Calculates the maximum retry attempts for the shared LlamaIndex LLM.

    Returns:
        The number of attempts (retries + 1), with a minimum of 1.
    """
    retries = int(getattr(settings.agents, "max_retries", 0))
    return max(1, retries + 1)


def _shared_llm_retries() -> int:
    """Calculates the configured retry count for LlamaIndex LLM backends.

    Returns:
        The number of retries, with a minimum of 0.
    """
    return max(0, int(getattr(settings.agents, "max_retries", 0)))


def _supports_native_llamaindex_retries(llm: Any) -> bool:
    """Determines if a LlamaIndex LLM instance supports native max_retries.

    Args:
        llm: The LLM instance to inspect.

    Returns:
        True if the LLM exposes a 'max_retries' field via Pydantic or legacy
        attribute naming.
    """
    fields = getattr(llm, "model_fields", None)
    if isinstance(fields, dict) and "max_retries" in fields:
        return True
    legacy_fields = getattr(llm, "__fields__", None)
    return isinstance(legacy_fields, dict) and "max_retries" in legacy_fields


class MultiAgentCoordinator:
    """Orchestrates five specialized agents for document analysis.

    Uses a graph-native LangGraph supervisor pattern to manage:
    - Router Agent: Determines query processing strategy.
    - Planner Agent: Decomposes complex queries.
    - Retrieval Agent: Executes document search with DSPy optimization.
    - Synthesis Agent: Combines multi-source results.
    - Validation Agent: Validates final response accuracy.

    Integrates context management, performance monitoring, and deadline
    propagation.
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
        """Initializes the multi-agent coordinator with runtime configurations.

        Args:
            model_path: The LLM model identifier.
            max_context_length: Maximum context window in tokens.
            backend: Inference backend identifier.
            enable_fallback: Whether to use basic RAG on agent failures.
            max_agent_timeout: Timeout threshold for agent decision per turn.
            tool_registry: Optional registry for resolving agent tools.
            use_shared_llm_client: Flag to reuse LlamaIndex LLM client wrappers.
            checkpointer: LangGraph checkpointer for state persistence.
            store: LangGraph BaseStore for long-term memory access.
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
        self._memory_consolidation_semaphore = threading.BoundedSemaphore(
            MEMORY_CONSOLIDATION_MAX_WORKERS
        )
        self._memory_executor_closed = False

        logger.info("MultiAgentCoordinator initialized (model: {})", model_path)

    def close(self) -> None:
        """Releases system resources, including the memory consolidation executor."""
        if self._memory_executor_closed:
            return
        self._memory_executor_closed = True
        with contextlib.suppress(Exception):
            self._memory_executor.shutdown(wait=False)

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        """Best-effort cleanup during interpreter teardown."""
        with contextlib.suppress(Exception):
            self.close()

    def _ensure_setup(self) -> bool:
        """Ensures all internal components are initialized via lazy loading.

        Returns:
            True if setup is successful or already complete, False on failure.
        """
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
                                "Configured shared LlamaIndex LLM retries: {}",
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
            redaction = build_pii_log_entry(str(e), key_id="coordinator.setup")
            logger.error(
                "Failed to setup coordinator (error_type={}, error={})",
                type(e).__name__,
                redaction.redacted,
            )
            return False

    def _setup_agent_graph(self) -> None:
        """Initializes the LangGraph supervisor and specialized worker nodes.

        Configures agent tools, middleware, and the supervisor system prompt.
        Compiled graph includes persistence support if checkpointer is provided.

        Raises:
            RuntimeError: If graph initialization or compilation fails.
        """
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

            hook_middleware = _AgentHookMiddleware(
                pre_model_hook=self._create_pre_model_hook(),
                post_model_hook=self._create_post_model_hook(),
            )

            agents: dict[str, Any] = {}
            for name, tool_loader in agent_specs:
                agents[name] = create_agent(
                    model,
                    tools=tool_loader(),
                    state_schema=MultiAgentGraphState,
                    middleware=[hook_middleware],
                    name=name,
                    store=self.store,
                )

            # Create supervisor system prompt
            system_prompt = self._create_supervisor_prompt()

            # Create list of agents for supervisor preserving definition order.
            supervisor_agents = [agents[name] for name, _ in agent_specs]

            forward_tool = create_forward_message_tool(supervisor_name="supervisor")

            # Graph-native supervisor (ADR-011): supervisor + subagents via
            # `StateGraph`, using handoff tools that return `Command.PARENT`.
            self.graph = build_multi_agent_supervisor_graph(
                supervisor_agents,
                model=model,
                prompt=system_prompt,
                state_schema=MultiAgentGraphState,
                middleware=[hook_middleware],
                extra_tools=[forward_tool],
                params=SupervisorBuildParams(
                    supervisor_name="supervisor",
                    output_mode="last_message",
                    add_handoff_messages=True,
                    add_handoff_back_messages=True,
                ),
            )

            # Store agents for reference
            self.agents = agents

            # Compile graph with checkpointer + optional store (ADR-058).
            self.compiled_graph = self.graph.compile(
                checkpointer=self.checkpointer, store=self.store
            )

            logger.info("Agent graph setup completed successfully")

        except (RuntimeError, ValueError, AttributeError) as e:
            redaction = build_pii_log_entry(str(e), key_id="coordinator.agent_graph")
            logger.error(
                "Failed to setup agent graph (error_type={}, error={})",
                type(e).__name__,
                redaction.redacted,
            )
            raise RuntimeError("Agent graph initialization failed") from e

    def _create_supervisor_prompt(self) -> str:
        """Generates the system prompt for the supervisor agent.

        Returns:
            A string containing role definitions, performance targets, and
            coordination strategy instructions.
        """
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
        """Creates a state-transformation hook for context trimming.

        Returns:
            A callable that estimating token counts and applies trimming logic
            before model execution.
        """

        def pre_model_hook(state: dict) -> dict:
            """Trims message history based on context window thresholds.

            Args:
                state: The current graph state.

            Returns:
                The modified state with trimmed messages if thresholds were exceeded.
            """
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
                redaction = build_pii_log_entry(str(e), key_id="coordinator.pre_hook")
                logger.warning(
                    "Pre-model hook failed (error_type={}, error={})",
                    type(e).__name__,
                    redaction.redacted,
                )
                # Non-fatal: annotate state for observability only if dict
                if isinstance(state, dict):
                    state["hook_error"] = True
                    state["hook_name"] = "pre_model_hook"
                return state

        return pre_model_hook

    def _create_post_model_hook(self) -> Callable[[dict], dict]:
        """Creates a state-transformation hook for metrics annotation.

        Returns:
            A callable that attaches optimization metrics to the state after
            model execution.
        """

        def post_model_hook(state: dict) -> dict:
            """Attaches token usage and KV cache metrics to the graph state.

            Args:
                state: The current graph state.

            Returns:
                The state augmented with optimization metrics.
            """
            try:
                # Build base metrics from current state via shared helper
                state["optimization_metrics"] = (
                    self._build_base_optimization_metrics_from_state(state)
                )
                return state
            except (RuntimeError, ValueError, AttributeError) as e:
                redaction = build_pii_log_entry(str(e), key_id="coordinator.post_hook")
                logger.warning(
                    "Post-model hook failed (error_type={}, error={})",
                    type(e).__name__,
                    redaction.redacted,
                )
                if isinstance(state, dict):
                    state["hook_error"] = True
                    state["hook_name"] = "post_model_hook"
                return state

        return post_model_hook

    def _build_base_optimization_metrics_from_state(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Derives primary optimization metrics from a graph state dictionary.

        Args:
            state: The graph state containing message history and status flags.

        Returns:
            A dictionary of base metrics including token counts, cache usage,
            and trimming status.
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
        """Consolidates all optimization metrics for the final response.

        Args:
            final_state: The terminal state of the agent workflow.
            coordination_time: Total duration of agent coordination in seconds.

        Returns:
            A dictionary of metrics including overhead and target compliance.
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
        """Records end-to-end latency and success rate via OpenTelemetry.

        Args:
            latency_s: Total processing time in seconds.
            success: Whether the operation completed without terminal errors.
        """
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
        """Starts a traced operation span with associated query attributes.

        Args:
            exit_stack: Context manager stack for automated lifecycle handling.
            thread_id: Unique conversation identifier.
            query_length: Character length of the input query.

        Returns:
            The active tracer span for the current processing context.
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
        """Constructs the starting state dictionary for the agent graph.

        Args:
            query: The processed user input string.
            start_time: Wall-clock start timestamp for the request.
            tools_data: Configuration and constraints for resolveable tools.

        Returns:
            A dictionary conforming to MultiAgentState schema.
        """
        deadline_ts = None
        if settings.agents.enable_deadline_propagation:
            deadline_ts = time.monotonic() + float(self.max_agent_timeout)
        return MultiAgentState(
            messages=[HumanMessage(content=query)],
            tools_data=tools_data,
            total_start_time=start_time,
            output_mode="last_message",
            parallel_execution_active=True,
            deadline_ts=deadline_ts,
        ).model_dump()

    def _maybe_semantic_cache_lookup(
        self,
        *,
        query: str,
        thread_id: str,
        user_id: str,
        checkpoint_id: str | None,
        start_time: float,
        span: Span,
    ) -> tuple[AgentResponse | None, CacheKey | None]:
        """Performs a fail-open semantic cache lookup for new conversation threads.

        Lookup is skipped if caching is disabled, history exists, or we are
        resuming from a specific checkpoint.

        Args:
            query: User input to match against the vector store.
            thread_id: Unique conversation identifier.
            user_id: Namespace for memory and scoping.
            checkpoint_id: Checkpoint ID if resuming (time travel).
            start_time: Timestamp for processing start.
            span: Active OpenTelemetry span.

        Returns:
            A tuple of (AgentResponse, CacheKey). Response is None on cache miss.
        """
        sem_cfg = getattr(settings, "semantic_cache", None)
        if sem_cfg is None or not bool(getattr(sem_cfg, "enabled", False)):
            return None, None
        if checkpoint_id is not None:
            return None, None

        try:
            has_history = bool(
                self.list_checkpoints(thread_id=thread_id, user_id=user_id, limit=1)
            )
        except (
            RuntimeError,
            ValueError,
            AttributeError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            redaction = build_pii_log_entry(
                str(exc), key_id="coordinator.history_check"
            )
            logger.warning(
                "History check failed (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            has_history = True
        if has_history:
            return None, None

        semantic_cache_key: CacheKey | None = None
        try:
            from src.persistence.hashing import compute_corpus_hash
            from src.persistence.langchain_embeddings import LlamaIndexEmbeddingsAdapter
            from src.persistence.snapshot_utils import collect_corpus_paths
            from src.utils.semantic_cache import (
                SemanticCache,
                build_cache_key,
                config_hash_for_semcache,
            )
            from src.utils.storage import create_sync_client

            uploads_dir = settings.data_dir / "uploads"
            corpus_hash = compute_corpus_hash(
                collect_corpus_paths(uploads_dir),
                base_dir=uploads_dir,
            )
            cfg_hash = config_hash_for_semcache(settings)

            model_id = str(settings.model or settings.vllm.model or "") or "unknown"
            semantic_cache_key = build_cache_key(
                query=str(query),
                namespace=str(settings.semantic_cache.namespace),
                model_id=model_id,
                template_id="chat",
                template_version=str(settings.app_version),
                temperature=float(settings.vllm.temperature),
                corpus_hash=corpus_hash,
                config_hash=cfg_hash,
            )

            with create_sync_client() as client:
                cache = SemanticCache(
                    client=client,
                    cfg=settings.semantic_cache,
                    vector_dim=int(settings.embedding.dimension),
                    embed_query=LlamaIndexEmbeddingsAdapter().embed_query,
                )
                hit = cache.lookup(key=semantic_cache_key, query=str(query))
        except Exception as exc:  # pragma: no cover - fail open
            redaction = build_pii_log_entry(
                str(exc), key_id="semantic_cache.lookup.exception"
            )
            logger.debug(
                "Semantic cache lookup skipped (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            return None, semantic_cache_key

        if hit is None:
            return None, semantic_cache_key

        processing_time = time.perf_counter() - start_time
        response = AgentResponse(
            content=str(hit.response_text),
            sources=[],
            metadata={
                "semantic_cache": {
                    "hit": True,
                    "kind": hit.kind,
                    "score": hit.score,
                }
            },
            validation_score=0.0,
            processing_time=processing_time,
            optimization_metrics={"semantic_cache": True},
            agent_decisions=[],
            fallback_used=False,
        )
        self.successful_queries += 1
        self._update_performance_metrics(processing_time, 0.0)
        self._record_query_metrics(processing_time, True)
        span.set_attribute("semantic_cache.hit", True)
        span.set_attribute("semantic_cache.kind", str(hit.kind))
        return response, semantic_cache_key

    def _maybe_semantic_cache_store(
        self,
        *,
        semantic_cache_key: CacheKey,
        query: str,
        response_text: str,
    ) -> None:
        """Stores a successful response in the semantic cache asynchronously.

        Args:
            semantic_cache_key: Validated cache key for storage.
            query: Original query string.
            response_text: The generated content to cache.
        """
        sem_cfg = getattr(settings, "semantic_cache", None)
        if sem_cfg is None or not bool(getattr(sem_cfg, "enabled", False)):
            return

        try:
            from src.persistence.langchain_embeddings import LlamaIndexEmbeddingsAdapter
            from src.utils.semantic_cache import SemanticCache
            from src.utils.storage import create_sync_client

            with create_sync_client() as client:
                cache = SemanticCache(
                    client=client,
                    cfg=settings.semantic_cache,
                    vector_dim=int(settings.embedding.dimension),
                    embed_query=LlamaIndexEmbeddingsAdapter().embed_query,
                )
                cache.store(
                    key=semantic_cache_key,
                    query=str(query),
                    response_text=str(response_text),
                )
        except Exception as exc:  # pragma: no cover - fail open
            redaction = build_pii_log_entry(
                str(exc), key_id="semantic_cache.store.exception"
            )
            logger.debug(
                "Semantic cache store skipped (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )

    def _handle_timeout_response(
        self, query: str, context: Any | None, start_time: float
    ) -> tuple[AgentResponse, bool]:
        """Orchestrates the response when the agent workflow exceeds its timeout.

        Applies basic RAG fallback if enabled, otherwise returns a standardized
        timeout error.

        Args:
            query: The original user query.
            context: Runtime context passed to the coordinator.
            start_time: Wall-clock start time for the entire request.

        Returns:
            A tuple of (AgentResponse, fallback_applied_flag).
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
                redaction = build_pii_log_entry(
                    str(exc), key_id="coordinator.fallback_metadata"
                )
                logger.debug(
                    "Failed to annotate fallback metadata (error_type={}, error={})",
                    type(exc).__name__,
                    redaction.redacted,
                )
            return response, True
        processing_time = time.perf_counter() - start_time
        response = AgentResponse(
            content=("The multi-agent system timed out while processing your request."),
            sources=[],
            metadata={"fallback_used": True, "reason": "timeout"},
            validation_score=0.0,
            processing_time=processing_time,
            optimization_metrics={"timeout": True},
        )
        return response, True

    def _handle_workflow_result(
        self,
        result: dict[str, Any] | None,
        query: str,
        context: Any | None,
        start_time: float,
        coordination_time: float,
    ) -> tuple[AgentResponse, bool, bool]:
        """Analyzes the agent workflow output and resolves it to a final response.

        Args:
            result: Terminal graph state or None on critical failure.
            query: Original query string.
            context: Runtime context.
            start_time: Processing start timestamp.
            coordination_time: Total accumulated overhead in coordination.

        Returns:
            A tuple of (AgentResponse, timed_out_flag, fallback_used_flag).
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
        """Updates internal performance averages following request completion.

        Args:
            workflow_timed_out: Indicates if the workflow exceeded limits.
            used_fallback: Indicates if a fallback strategy was invoked.
            processing_time: Total request duration in seconds.
            coordination_time: coordination duration in seconds.
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
        """Attaches final outcome attributes to the OpenTelemetry trace span.

        Args:
            span: The span to annotate.
            workflow_timed_out: Final timeout status.
            used_fallback: Final fallback status.
            processing_time: Terminal duration in seconds.
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
        """Executes a document analysis request through the multi-agent pipeline.

        Orchestrates specialized agents, manages context constraints, and
        provides durability via LangGraph checkpoints.

        Args:
            query: The user query to process.
            context: Caller-provided runtime context (transient).
            settings_override: Dictionary of configuration overrides.
            thread_id: Identifier for cross-request conversation state.
            user_id: Namespace for memory and policy scoping.
            checkpoint_id: Optional ID to resume from a specific state.

        Returns:
            An AgentResponse containing content, sources, and metrics.
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
                cached_response, semantic_cache_key = self._maybe_semantic_cache_lookup(
                    query=query,
                    thread_id=thread_id,
                    user_id=user_id,
                    checkpoint_id=checkpoint_id,
                    start_time=start_time,
                    span=span,
                )
                if cached_response is not None:
                    return cached_response

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

                # Best-effort semantic cache store (never impact user flow).
                if (
                    semantic_cache_key is not None
                    and not workflow_timed_out
                    and not used_fallback
                ):
                    self._maybe_semantic_cache_store(
                        semantic_cache_key=semantic_cache_key,
                        query=str(query),
                        response_text=str(response.content),
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
                redaction = build_pii_log_entry(
                    str(exc), key_id="coordinator.process_query.exception"
                )
                span.set_attribute("coordinator.error", redaction.redacted)
                span.set_attribute("coordinator.error_type", type(exc).__name__)
                logger.error(
                    "Multi-agent processing failed (error_type={}, error={})",
                    type(exc).__name__,
                    redaction.redacted,
                )

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
        """Executes the agent graph with wall-clock timeout protection.

        Args:
            initial_state: The starting dictionary for graph execution.
            thread_id: Persistence thread identifier.
            user_id: Persistence namespace for memory.
            checkpoint_id: Optional ID to resume from a prior state.
            runtime_context: Key-value pairs injected into tool execution.

        Returns:
            The terminal state dictionary. 'timed_out' flag is set if the
            decision timeout is exceeded.
        """
        try:
            if self.compiled_graph is None:
                raise RuntimeError("Agent graph is not compiled")

            deadline_ts = (
                _coerce_deadline_ts(initial_state)
                if settings.agents.enable_deadline_propagation
                else None
            )

            _ensure_event_loop()

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
                result = _as_state_dict(state)
                elapsed = time.perf_counter() - float(
                    initial_state.get("total_start_time", 0.0)
                )

                if deadline_ts is not None and time.monotonic() > deadline_ts:
                    logger.warning(
                        "Agent workflow deadline exceeded after {:.2f}s", elapsed
                    )
                    result["timed_out"] = True
                    result["deadline_s"] = float(self.max_agent_timeout)
                    result["cancel_reason"] = "deadline_exceeded"
                    with contextlib.suppress(Exception):  # pragma: no cover - telemetry
                        log_jsonl(
                            {
                                "agent_deadline_exceeded": True,
                                "decision_timeout_s": float(self.max_agent_timeout),
                                "elapsed_s": float(elapsed),
                            }
                        )
                    break

                if deadline_ts is None and elapsed > self.max_agent_timeout:
                    logger.warning("Agent workflow timeout after {:.2f}s", elapsed)
                    result["timed_out"] = True
                    result["deadline_s"] = float(self.max_agent_timeout)
                    break

            if result is not None:
                return result
            thread_redacted = build_pii_log_entry(
                str(thread_id), key_id="coordinator.thread_id"
            ).redacted
            logger.warning(
                "Agent workflow produced no result; returning initial "
                "state (thread_id={})",
                thread_redacted,
            )
            return initial_state

        except (RuntimeError, ValueError, AttributeError, TimeoutError) as e:
            redaction = build_pii_log_entry(str(e), key_id="agent_workflow.exception")
            logger.error(
                "Agent workflow execution failed (error_type={}, error={})",
                type(e).__name__,
                redaction.redacted,
            )
            raise

    def _extract_response(
        self,
        final_state: dict[str, Any],
        _original_query: str,
        start_time: float,
        coordination_time: float,
    ) -> AgentResponse:
        """Parses the terminal graph state into a structured AgentResponse.

        Args:
            final_state: The terminal state dictionary from graph execution.
            _original_query: The user input query.
            start_time: Wall-clock processing start time.
            coordination_time: Overhead time spent coordinating agents.

        Returns:
            An AgentResponse object containing content, sources, and metrics.
        """
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
                    "framework": "LangGraph StateGraph Supervisor",
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
            redaction = build_pii_log_entry(
                str(e), key_id="coordinator.extract_response"
            )
            logger.error(
                "Failed to extract response (error_type={}, error={})",
                type(e).__name__,
                redaction.redacted,
            )
            processing_time = time.perf_counter() - start_time
            return AgentResponse(
                content="Error extracting response.",
                sources=[],
                metadata={
                    "extraction_error_type": type(e).__name__,
                    "extraction_error": redaction.redacted,
                },
                validation_score=0.0,
                processing_time=processing_time,
                optimization_metrics={"error": True},
            )

    def _create_error_response(
        self, error_msg: str, start_time: float
    ) -> AgentResponse:
        """Generates a standardized error response with timing metadata.

        Args:
            error_msg: Descriptive error message.
            start_time: Processing start timestamp.

        Returns:
            An AgentResponse indicating initialization or execution failure.
        """
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
        """Executes a basic response strategy when the multi-agent graph fails.

        Args:
            query: The original user query.
            _context: Transient runtime context.
            start_time: Processing start timestamp.

        Returns:
            A simplified AgentResponse indicating system unavailability.
        """
        try:
            # Update fallback counter (production behavior)
            self.fallback_queries += 1
            logger.info("Using basic RAG fallback")

            # Simple fallback response
            processing_time = time.perf_counter() - start_time

            return AgentResponse(
                content=(
                    "The multi-agent system is temporarily unavailable. "
                    "Please try again shortly."
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
            redaction = build_pii_log_entry(str(e), key_id="coordinator.fallback_rag")
            logger.error(
                "Fallback RAG also failed (error_type={}, error={})",
                type(e).__name__,
                redaction.redacted,
            )
            processing_time = time.perf_counter() - start_time
            return AgentResponse(
                content="System temporarily unavailable.",
                sources=[],
                metadata={
                    "fallback_used": True,
                    "fallback_failed": True,
                    "error_type": type(e).__name__,
                    "error": redaction.redacted,
                },
                validation_score=0.0,
                processing_time=processing_time,
                optimization_metrics={"system_failure": True},
            )

    def _update_performance_metrics(
        self, processing_time: float, coordination_time: float
    ) -> None:
        """Updates cumulative averages for total duration and agent overhead.

        Args:
            processing_time: Total duration of the current query in seconds.
            coordination_time: Time spent in supervisor decision turns.
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
        """Aggregates cumulative performance and ADR compliance statistics.

        Returns:
            A dictionary containing success rates, timing averages, and
            compliance flags.
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
        """Retrieves persisted state values for a specific conversation thread.

        Args:
            thread_id: Unique conversation identifier.
            user_id: User namespace for scoping.
            checkpoint_id: Optional ID for a non-terminal checkpoint.

        Returns:
            A dictionary of state values, or empty if not found.
        """
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
        """Lists historical checkpoints for a thread in reverse-chronological order.

        Args:
            thread_id: Unique conversation identifier.
            user_id: User namespace for scoping.
            limit: Maximum number of checkpoints to retrieve.

        Returns:
            A list of dictionary objects containing checkpoint metadata.
        """
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
            redaction = build_pii_log_entry(
                str(exc), key_id="coordinator.list_checkpoints"
            )
            logger.debug(
                "Checkpoint listing failed (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
        return out

    def fork_from_checkpoint(
        self,
        *,
        thread_id: str,
        user_id: str = "local",
        checkpoint_id: str,
    ) -> str | None:
        """Forks a conversation branch from a historical checkpoint.

        Creates a new head in the checkpoint store based on a prior state,
        enabling non-destructive exploration (time-travel).

        Args:
            thread_id: Unique conversation identifier.
            user_id: User namespace for scoping.
            checkpoint_id: The source checkpoint identifier to fork from.

        Returns:
            The new checkpoint ID of the forked branch, or None on failure.
        """
        if not self._ensure_setup() or self.compiled_graph is None:
            return None

        cfg: dict[str, Any] = {
            "thread_id": str(thread_id),
            "user_id": str(user_id),
            "checkpoint_id": str(checkpoint_id),
        }
        config: RunnableConfig = {"configurable": cfg}
        thread_redacted = build_pii_log_entry(
            str(thread_id), key_id="coordinator.thread_id"
        ).redacted
        checkpoint_redacted = build_pii_log_entry(
            str(checkpoint_id), key_id="coordinator.checkpoint_id"
        ).redacted

        # First, fetch the checkpoint to verify it exists and preserve metadata
        try:
            source_state = self.compiled_graph.get_state(config)
            if not source_state or not source_state.values:
                logger.debug(
                    "Checkpoint not found or empty (thread_id={} checkpoint_id={})",
                    thread_redacted,
                    checkpoint_redacted,
                )
                return None
        except (RuntimeError, ValueError, AttributeError) as exc:
            err = build_pii_log_entry(str(exc), key_id="coordinator.get_state")
            logger.debug(
                "Failed to fetch checkpoint (thread_id={} "
                "checkpoint_id={} error_type={} error={})",
                thread_redacted,
                checkpoint_redacted,
                type(exc).__name__,
                err.redacted,
            )
            return None

        # Now fork the checkpoint using update_state with __copy__ node
        try:
            # LangGraph special update that clones the selected checkpoint into a
            # new head (time-travel fork) without running the workflow.
            # Note: as_node="__copy__" is LangGraph 0.2.0+ API for
            # metadata preservation.
            new_config = self.compiled_graph.update_state(
                config, None, as_node="__copy__"
            )
        except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
            # Specific exception handling for LangGraph API errors
            err = build_pii_log_entry(str(exc), key_id="coordinator.update_state")
            logger.debug(
                "Checkpoint fork failed (thread_id={} checkpoint_id={} "
                "error_type={} error={})",
                thread_redacted,
                checkpoint_redacted,
                type(exc).__name__,
                err.redacted,
            )
            return None
        except Exception as exc:
            # Catch any unexpected exceptions but log them for debugging
            err = build_pii_log_entry(
                str(exc), key_id="coordinator.update_state.unhandled"
            )
            logger.debug(
                "Unexpected error during checkpoint fork (thread_id={} "
                "checkpoint_id={} error_type={} error={})",
                thread_redacted,
                checkpoint_redacted,
                type(exc).__name__,
                err.redacted,
            )
            return None

        conf = new_config.get("configurable") if isinstance(new_config, dict) else None
        conf = conf if isinstance(conf, dict) else {}
        new_checkpoint_id = conf.get("checkpoint_id")
        return str(new_checkpoint_id) if new_checkpoint_id else None

    def validate_system_status(self) -> dict[str, bool]:
        """Validates the operational status and requirements of all components.

        Returns:
            A dictionary of boolean flags indicating component health.
        """
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
        """Resets all cumulative performance metrics to zero."""
        self.total_queries = 0
        self.successful_queries = 0
        self.fallback_queries = 0
        self.avg_processing_time = 0.0
        self.avg_coordination_overhead = 0.0
        logger.info("Performance statistics reset")

    def validate_adr_compliance(self) -> dict[str, bool]:
        """Evaluates architectural requirements against current system state.

        Returns:
            A dictionary of boolean flags for specific ADR requirements.
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
        """Executes background memory consolidation for a completed turn.

        Extracts potential memory candidates from history and merges them into
        the long-term store based on configured similarity and importance
        policies.

        Args:
            final_state: The terminal state of the agent graph.
            thread_id: Unique conversation identifier.
            user_id: User namespace for memory scoping.
        """
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
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(
                str(exc), key_id="coordinator.memory_consolidation_background"
            )
            logger.debug(
                "Memory consolidation background task failed (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )

    def _schedule_memory_consolidation(
        self,
        final_state: dict[str, Any],
        *,
        thread_id: str,
        user_id: str,
    ) -> None:
        """Schedules memory consolidation to run off the critical path.

        Args:
            final_state: The terminal state of the agent graph.
            thread_id: Unique conversation identifier.
            user_id: User namespace for memory scoping.
        """
        if self.store is None:
            return
        if not self._memory_consolidation_semaphore.acquire(blocking=False):
            logger.debug(
                "Skipping memory consolidation; max in-flight reached for thread {}",
                thread_id,
            )
            return

        future = self._memory_executor.submit(
            self._consolidate_memories, final_state, thread_id, user_id
        )

        def _release_slot() -> None:
            with contextlib.suppress(ValueError):
                self._memory_consolidation_semaphore.release()

        def _watch_future() -> None:
            should_release = True
            released = threading.Event()

            def _release_once(reason: str) -> None:
                if released.is_set():
                    return
                released.set()
                logger.debug(
                    "Releasing memory consolidation slot ({}) for thread {}",
                    reason,
                    thread_id,
                )
                _release_slot()

            def _release_if_stuck() -> None:
                if future.done():
                    return
                logger.warning(
                    "Memory consolidation still running after timeout+grace "
                    "for thread {}",
                    thread_id,
                )
                _release_once("grace-timeout")

            try:
                future.result(timeout=MEMORY_CONSOLIDATION_TIMEOUT_S)
            except FuturesTimeoutError:
                logger.warning(
                    "Memory consolidation timed out after {}s for thread {}",
                    MEMORY_CONSOLIDATION_TIMEOUT_S,
                    thread_id,
                )
                if not future.cancel():
                    logger.debug(
                        "Memory consolidation still running after timeout for "
                        "thread {}",
                        thread_id,
                    )

                    # Attach callback to release slot when future eventually completes.
                    # This prevents semaphore leak when task doesn't respond to cancel.
                    def _on_future_done(fut: Any) -> None:
                        with contextlib.suppress(Exception):
                            fut.result()  # Log any exception but don't propagate
                        _release_once("done")

                    try:
                        future.add_done_callback(_on_future_done)
                        should_release = False  # Callback/timer will handle release
                        threading.Timer(
                            MEMORY_CONSOLIDATION_RELEASE_GRACE_S, _release_if_stuck
                        ).start()
                    except (RuntimeError, TypeError) as exc:
                        from src.utils.log_safety import build_pii_log_entry

                        redaction = build_pii_log_entry(
                            str(exc),
                            key_id="coordinator.memory_consolidation_add_callback",
                        )
                        # If callback cannot be added (rare; future may be done),
                        # release immediately as fallback
                        logger.debug(
                            "Could not attach done callback for thread {} "
                            "(error_type={} error={}); releasing slot immediately",
                            thread_id,
                            type(exc).__name__,
                            redaction.redacted,
                        )
                        _release_once("callback-failed")
                        should_release = False  # Already released, avoid double-release
            except Exception as exc:
                from src.utils.log_safety import build_pii_log_entry

                redaction = build_pii_log_entry(
                    str(exc), key_id="coordinator.memory_consolidation_task"
                )
                logger.debug(
                    "Memory consolidation task failed (error_type={} error={})",
                    type(exc).__name__,
                    redaction.redacted,
                )
            finally:
                if should_release:
                    _release_once("completed")

        threading.Thread(
            target=_watch_future,
            name="docmind-memory-watch",
            daemon=True,
        ).start()


# Factory function for coordinator
def create_multi_agent_coordinator(
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507-FP8",
    max_context_length: int = settings.vllm.context_window,
    enable_fallback: bool = True,
    *,
    tool_registry: ToolRegistry | None = None,
    use_shared_llm_client: bool | None = None,
) -> MultiAgentCoordinator:
    """Factory function for initializing the MultiAgentCoordinator.

    Args:
        model_path: The LLM model identifier.
        max_context_length: Maximum context window in tokens.
        enable_fallback: Whether to use basic RAG on agent failures.
        tool_registry: Optional registry for resolving agent tools.
        use_shared_llm_client: Flag to reuse LlamaIndex LLM client wrappers.

    Returns:
        A fully configured MultiAgentCoordinator instance.
    """
    return MultiAgentCoordinator(
        model_path=model_path,
        max_context_length=max_context_length,
        enable_fallback=enable_fallback,
        tool_registry=tool_registry,
        use_shared_llm_client=use_shared_llm_client,
    )
