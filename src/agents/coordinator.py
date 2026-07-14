"""Multi-Agent Coordination System using a graph-native LangGraph supervisor.

This module implements a MultiAgentCoordinator that orchestrates four specialized agents
using a graph-native LangGraph `StateGraph` supervisor for query processing, with
deadline propagation, runtime context injection, and checkpoint/store support.

Features:
- LangGraph `StateGraph` supervisor orchestration
- Structured output mode for consistent response formatting
- Atomic multi-agent dispatch
- Context trimming hooks for memory management
- Performance tracking and timeout protection

Example:
    Using the multi-agent coordinator:

        from src.agents.coordinator import MultiAgentCoordinator
        coordinator = MultiAgentCoordinator()
        response = coordinator.process_query("Compare AI vs ML techniques")
        # response.content contains the generated text
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import sqlite3
import threading
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import CancelledError as FuturesCancelledError
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphDrained
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import RunControl

# Import LlamaIndex Settings for context window access
from llama_index.core.utils import get_tokenizer
from loguru import logger
from opentelemetry import metrics, trace
from opentelemetry.trace import Span

from src.agents.registry import build_agent_tool_sets

# Graph-native supervisor implementation (repo-local).
from src.agents.supervisor_graph import (
    SupervisorBuildParams,
    build_multi_agent_supervisor_graph,
)
from src.agents.tools.constants import MAX_RETRIEVAL_RESULTS
from src.agents.tools.memory import (
    MemoryConsolidationPolicy,
    capture_memory_namespace_generations,
    consolidate_and_apply_memory_candidates,
    extract_memory_candidates,
    is_memory_namespace_tombstoned,
    memory_generation_from_state,
    memory_namespace_generation,
    memory_namespace_lock,
    try_advance_memory_namespace_generation,
    try_tombstone_memory_namespace,
)
from src.agents.tools.synthesis import (
    current_retrieval_batches,
    interleave_retrieval_documents,
    retrieval_batch_watermark,
)
from src.config import settings
from src.config.langchain_factory import build_chat_model
from src.persistence.chat_db import (
    CHECKPOINT_IDENTITY_TABLES,
    INCOMPATIBLE_CHECKPOINT_DB_MESSAGE,
    LEGACY_CHECKPOINT_IDENTITY_MESSAGE,
    LEGACY_CHECKPOINT_IDENTITY_WHERE,
    LegacyCheckpointIdentityError,
)
from src.persistence.chat_db import purge_session as delete_persisted_session
from src.persistence.checkpoint_identity import checkpoint_thread_id, memory_namespace
from src.telemetry.opentelemetry import configure_observability
from src.utils.log_safety import build_pii_log_entry
from src.utils.telemetry import log_jsonl

# Import agent-specific models
from .models import (
    AgentResponse,
    AgentRuntimeContext,
    MultiAgentGraphState,
    MultiAgentState,
)

_COORDINATOR_TRACER = trace.get_tracer("docmind.agents.coordinator")
_COORDINATOR_LATENCY = None
_COORDINATOR_COUNTER = None


class ContextManager:
    """Manage context-window constraints and token estimation."""

    def __init__(self, max_context_tokens: int = 131072) -> None:
        """Initializes the context manager with specified token limits.

        Args:
            max_context_tokens: Maximum allowed tokens in the context window.
        """
        self.max_context_tokens = max_context_tokens
        self.trim_threshold = int(max_context_tokens * 0.9)  # 90% threshold
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


# Constants
COORDINATION_OVERHEAD_THRESHOLD = 0.2  # seconds (200ms target)
CONTEXT_TRIM_STRATEGY = "last"
MEMORY_CONSOLIDATION_MAX_WORKERS = 2
MEMORY_CONSOLIDATION_TIMEOUT_S = 10.0
AGENT_GRAPH_MAX_CONCURRENT_RUNS = 4
AGENT_GRAPH_RUNNER_CLOSE_GRACE_S = 1.0
SESSION_PURGE_DRAIN_TIMEOUT_S = 5.0
_TIMEOUT_REASONS = frozenset({"deadline_exceeded", "dependency_timeout"})
_T = TypeVar("_T")


class _GraphRunnerCapacityError(RuntimeError):
    """Raised when the bounded graph runner has no execution slot available."""


class _DaemonTaskExecutor:
    """Start admitted background work on daemon threads without an exit join."""

    def __init__(self, *, thread_name: str) -> None:
        self._thread_name = thread_name
        self._lock = threading.Lock()
        self._closed = False

    def submit(
        self,
        function: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[_T]:
        """Start one already-capacity-checked task immediately."""
        future: Future[_T] = Future()

        def _run() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        with self._lock:
            if self._closed:
                raise RuntimeError("Memory executor is closed")
            threading.Thread(
                target=_run,
                name=self._thread_name,
                daemon=True,
            ).start()
        return future

    def close(self) -> None:
        """Reject new work without joining admitted daemon tasks."""
        with self._lock:
            self._closed = True


class _AsyncGraphRunner:
    """Own one persistent event loop for bounded asynchronous graph execution."""

    def __init__(self, max_concurrent: int = AGENT_GRAPH_MAX_CONCURRENT_RUNS) -> None:
        self._loop = asyncio.new_event_loop()
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._lock = threading.Lock()
        self._closed = False
        self._async_cleanup: Callable[[], Coroutine[Any, Any, None]] | None = None
        self._thread = threading.Thread(
            target=self._run_loop,
            name="docmind-agent-graph",
            daemon=True,
        )
        try:
            self._thread.start()
        except RuntimeError:
            self._loop.close()
            raise

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                _, pending = self._loop.run_until_complete(
                    asyncio.wait(
                        pending,
                        timeout=AGENT_GRAPH_RUNNER_CLOSE_GRACE_S,
                    )
                )
                if pending:
                    logger.warning(
                        "Agent graph runner abandoned {} cancellation-resistant tasks",
                        len(pending),
                    )
            if self._async_cleanup is not None:
                cleanup_task = self._loop.create_task(self._async_cleanup())
                try:
                    done, cleanup_pending = self._loop.run_until_complete(
                        asyncio.wait(
                            {cleanup_task},
                            timeout=AGENT_GRAPH_RUNNER_CLOSE_GRACE_S,
                        )
                    )
                    if cleanup_pending:
                        cleanup_task.cancel()
                        logger.warning(
                            "Agent graph runner cleanup exceeded close grace period"
                        )
                    elif done:
                        cleanup_task.result()
                except Exception as exc:  # pragma: no cover - best-effort teardown
                    logger.warning(
                        "Agent graph runner cleanup failed (error_type={})",
                        type(exc).__name__,
                    )
            shutdown_task = self._loop.create_task(self._loop.shutdown_asyncgens())
            _, shutdown_pending = self._loop.run_until_complete(
                asyncio.wait(
                    {shutdown_task},
                    timeout=AGENT_GRAPH_RUNNER_CLOSE_GRACE_S,
                )
            )
            if shutdown_pending:
                shutdown_task.cancel()
                logger.warning("Async generator shutdown exceeded close grace period")
            remaining = asyncio.all_tasks(self._loop)
            for task in remaining:
                task.cancel()
                # The loop is closing by contract; suppress misleading destructor
                # noise for cancellation-resistant third-party coroutines.
                task._log_destroy_pending = False  # type: ignore[attr-defined]
            self._loop.close()

    async def _run_with_slot(
        self, coroutine: Coroutine[Any, Any, dict[str, Any]]
    ) -> dict[str, Any]:
        try:
            return await coroutine
        finally:
            self._slots.release()

    def submit(
        self, coroutine: Coroutine[Any, Any, dict[str, Any]]
    ) -> Future[dict[str, Any]]:
        """Submit one graph run without allowing unbounded queued work."""
        if not self._slots.acquire(blocking=False):
            coroutine.close()
            raise _GraphRunnerCapacityError("Agent graph runner is at capacity")

        with self._lock:
            if self._closed:
                self._slots.release()
                coroutine.close()
                raise RuntimeError("Agent graph runner is closed")

            wrapped = self._run_with_slot(coroutine)
            try:
                return asyncio.run_coroutine_threadsafe(wrapped, self._loop)
            except BaseException:
                wrapped.close()
                coroutine.close()
                self._slots.release()
                raise

    def run(self, coroutine: Coroutine[Any, Any, _T]) -> _T:
        """Run lifecycle work on the owned event loop and return its result."""
        with self._lock:
            if self._closed:
                coroutine.close()
                raise RuntimeError("Agent graph runner is closed")
            if threading.current_thread() is self._thread:
                coroutine.close()
                raise RuntimeError("Cannot synchronously wait on the graph runner")
            try:
                future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
            except BaseException:
                coroutine.close()
                raise
        return future.result()

    @property
    def is_alive(self) -> bool:
        """Return whether the owned event-loop thread is still running."""
        return self._thread.is_alive()

    def close(
        self,
        *,
        async_cleanup: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        """Cancel pending graph tasks and stop the event loop within a fixed grace."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._async_cleanup = async_cleanup
            if not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._loop.stop)

        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=AGENT_GRAPH_RUNNER_CLOSE_GRACE_S)
        if self._thread.is_alive():
            logger.warning("Agent graph runner did not stop within close grace period")


@dataclass(slots=True)
class _ActiveAgentRun:
    """Run-scoped cancellation and cleanup state for one conversation thread."""

    control: RunControl
    finished: threading.Event


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


def _require_deadline_ts(initial_state: dict[str, Any]) -> float:
    """Return the required finite monotonic deadline from graph state.

    Args:
        initial_state: The initial state dictionary containing potential deadlines.

    Returns:
        The absolute monotonic deadline.

    Raises:
        ValueError: If the deadline is absent, non-numeric, or non-finite.
    """
    raw_deadline = initial_state.get("deadline_ts")
    if raw_deadline is None:
        raise ValueError("deadline_ts must be a finite monotonic timestamp")
    try:
        deadline_ts = float(raw_deadline)
    except (TypeError, ValueError) as exc:
        raise ValueError("deadline_ts must be a finite monotonic timestamp") from exc
    if not math.isfinite(deadline_ts):
        raise ValueError("deadline_ts must be a finite monotonic timestamp")
    return deadline_ts


def _checkpoint_config(
    *,
    thread_id: str,
    user_id: str,
    checkpoint_id: str | None = None,
) -> RunnableConfig:
    """Build the sole user-scoped LangGraph persistence configuration."""
    configurable: dict[str, Any] = {
        "thread_id": checkpoint_thread_id(
            thread_id=str(thread_id),
            user_id=str(user_id),
        ),
        "public_thread_id": str(thread_id),
        "user_id": str(user_id),
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = str(checkpoint_id)
    return {"configurable": configurable}


async def _open_async_sqlite_checkpointer(
    path: Path,
) -> tuple[AsyncSqliteSaver, contextlib.AsyncExitStack]:
    """Open and initialize an async SQLite saver on the current event loop."""
    stack = contextlib.AsyncExitStack()
    try:
        saver = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(path))
        )
        await saver.setup()
        for table in CHECKPOINT_IDENTITY_TABLES:
            async with saver.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ) as cursor:
                if await cursor.fetchone() is None:
                    continue
            try:
                async with saver.conn.execute(
                    f"""
                    SELECT 1
                    FROM {table}
                    WHERE {LEGACY_CHECKPOINT_IDENTITY_WHERE}
                    LIMIT 1;
                    """,  # noqa: S608 - static internal table allowlist
                ) as cursor:
                    incompatible = await cursor.fetchone()
            except sqlite3.DatabaseError as exc:
                raise LegacyCheckpointIdentityError(
                    INCOMPATIBLE_CHECKPOINT_DB_MESSAGE
                ) from exc
            if incompatible is not None:
                raise LegacyCheckpointIdentityError(LEGACY_CHECKPOINT_IDENTITY_MESSAGE)
    except BaseException:
        await stack.aclose()
        raise
    return saver, stack


class MultiAgentCoordinator:
    """Orchestrates four specialized agents for document analysis.

    Uses a graph-native LangGraph supervisor pattern to manage:
    - Planner Agent: Decomposes complex queries.
    - Retrieval Agent: Executes the native retrieval router.
    - Synthesis Agent: Combines multi-source results.
    - Validation Agent: Validates final response accuracy.

    Integrates context management, performance monitoring, and deadline
    propagation.
    """

    def __init__(
        self,
        *,
        max_agent_timeout: float | None = None,
        checkpointer: Any | None = None,
        checkpointer_path: Path | None = None,
        store: Any | None = None,
    ):
        """Initializes the multi-agent coordinator with runtime configurations.

        Args:
            max_agent_timeout: Optional per-turn timeout override.
            checkpointer: LangGraph checkpointer for state persistence.
            checkpointer_path: SQLite path for a coordinator-owned async saver.
            store: LangGraph BaseStore for long-term memory access.

        Raises:
            ValueError: If both a checkpointer and checkpointer path are supplied.
        """
        configure_observability(settings)
        self.max_agent_timeout = float(
            settings.agents.decision_timeout
            if max_agent_timeout is None
            else max_agent_timeout
        )
        if self.max_agent_timeout <= 0:
            raise ValueError("max_agent_timeout must be greater than zero")
        if checkpointer is not None and checkpointer_path is not None:
            raise ValueError(
                "checkpointer and checkpointer_path are mutually exclusive"
            )

        # Context management with unified settings
        self.context_manager = ContextManager(
            max_context_tokens=settings.effective_context_window
        )

        # Checkpointer + store (ADR-058). Defaults to in-memory for tests.
        self._checkpointer_path = (
            Path(checkpointer_path) if checkpointer_path is not None else None
        )
        if checkpointer is not None:
            self.checkpointer = checkpointer
        elif self._checkpointer_path is not None:
            self.checkpointer = None
        else:
            self.checkpointer = InMemorySaver()
        self._checkpointer_stack: contextlib.AsyncExitStack | None = None
        self._checkpointer_lock = threading.Lock()
        self._persistence_lock = threading.Lock()
        self.store = store

        # Initialize components
        self.llm = None
        self.compiled_graph = None
        self.graph = None
        self.agents = {}

        # Lazy initialization
        self._setup_lock = threading.Lock()
        self._setup_complete = False
        self._memory_executor = _DaemonTaskExecutor(
            thread_name="docmind-memory",
        )
        self._memory_consolidation_semaphore = threading.BoundedSemaphore(
            MEMORY_CONSOLIDATION_MAX_WORKERS
        )
        self._memory_executor_closed = False
        self._memory_jobs_lock = threading.Lock()
        self._memory_jobs: dict[tuple[str, ...], int] = {}
        self._graph_runner: _AsyncGraphRunner | None = None
        self._graph_runner_lock = threading.Lock()
        self._active_runs: dict[str, _ActiveAgentRun] = {}
        self._purged_persistence_ids: set[str] = set()
        self._active_runs_lock = threading.Lock()
        self._closed = False

        logger.info(
            "MultiAgentCoordinator initialized (model: {})", settings.effective_model
        )

    def close(self) -> None:
        """Release background graph and memory resources."""
        with self._graph_runner_lock:
            with self._memory_jobs_lock:
                if self._closed:
                    return
                self._closed = True
                memory_namespaces = set(self._memory_jobs)
            graph_runner = self._graph_runner
            self._graph_runner = None
            checkpointer_stack = self._checkpointer_stack
            self._checkpointer_stack = None

        if not self._memory_executor_closed:
            self._memory_executor_closed = True
            with contextlib.suppress(Exception):
                self._memory_executor.close()

        for namespace in memory_namespaces:
            # Invalidate work that has not crossed the final mutation boundary.
            # An already-admitted synchronous commit is uncancellable, so bounded
            # shutdown logs and skips instead of waiting on its namespace lock.
            if try_advance_memory_namespace_generation(namespace) is None:
                logger.warning(
                    "Memory generation invalidation skipped during bounded shutdown; "
                    "an admitted mutation is still running"
                )
        with self._active_runs_lock:
            active_runs = tuple(self._active_runs.values())
        for active_run in active_runs:
            active_run.control.request_drain("coordinator_closed")

        if graph_runner is not None:
            cleanup = (
                checkpointer_stack.aclose if checkpointer_stack is not None else None
            )
            # Closing must remain bounded even if setup or a synchronous
            # persistence bridge is stuck. Saver ownership was captured atomically
            # with the runner above; late setup publication now fails closed.
            with contextlib.suppress(Exception):
                graph_runner.close(async_cleanup=cleanup)

    def _get_graph_runner(self) -> _AsyncGraphRunner:
        """Return the lazily created event-loop runner owned by this coordinator."""
        with self._graph_runner_lock:
            if self._closed:
                raise RuntimeError("MultiAgentCoordinator is closed")
            if self._graph_runner is None:
                self._graph_runner = _AsyncGraphRunner()
            return self._graph_runner

    def _ensure_checkpointer(self) -> None:
        """Create the configured async SQLite saver on its owning runner loop."""
        if self.checkpointer is not None:
            return
        if self._checkpointer_path is None:
            raise RuntimeError("No checkpointer configuration is available")

        with self._checkpointer_lock:
            if self.checkpointer is not None:
                return
            runner: _AsyncGraphRunner | None = None
            stack: contextlib.AsyncExitStack | None = None
            try:
                self._checkpointer_path.parent.mkdir(parents=True, exist_ok=True)
                runner = _AsyncGraphRunner()
                saver, stack = runner.run(
                    _open_async_sqlite_checkpointer(self._checkpointer_path)
                )
                with self._graph_runner_lock:
                    if self._closed:
                        raise RuntimeError("MultiAgentCoordinator is closed")
                    if self._graph_runner is not None:
                        raise RuntimeError(
                            "Graph runner initialized before owned checkpointer"
                        )
                    self._graph_runner = runner
                    self.checkpointer = saver
                    self._checkpointer_stack = stack
                return
            except BaseException as exc:
                if stack is not None and runner is not None:
                    with contextlib.suppress(Exception):
                        runner.run(stack.aclose())
                if runner is not None:
                    with contextlib.suppress(Exception):
                        runner.close()
                if isinstance(exc, Exception):
                    raise RuntimeError(
                        "Async SQLite checkpointer initialization failed"
                    ) from exc
                raise

    def _begin_active_run(self, persistence_id: str) -> _ActiveAgentRun | None:
        """Fence concurrent graph mutation for one persisted conversation thread."""
        with self._active_runs_lock:
            if (
                persistence_id in self._purged_persistence_ids
                or persistence_id in self._active_runs
            ):
                return None
            active_run = _ActiveAgentRun(RunControl(), threading.Event())
            self._active_runs[persistence_id] = active_run
            return active_run

    def _finish_active_run(
        self,
        persistence_id: str,
        active_run: _ActiveAgentRun,
    ) -> None:
        """Release a thread fence only after the async graph wrapper has exited."""
        with self._active_runs_lock:
            if self._active_runs.get(persistence_id) is active_run:
                self._active_runs.pop(persistence_id, None)
        active_run.finished.set()

    def _ensure_setup(self) -> bool:
        """Ensures all internal components are initialized via lazy loading.

        Returns:
            True if setup is successful or already complete, False on failure.
        """
        with self._setup_lock:
            with self._graph_runner_lock:
                if self._closed:
                    return False
                if self._setup_complete:
                    return True

            try:
                # Initialize vLLM environment for FP8 optimization
                from src.config import setup_llamaindex

                setup_llamaindex()
                logger.info("vLLM configuration applied via unified settings")

                # Use LLM from unified configuration (LlamaIndex Settings)
                from llama_index.core import Settings

                if getattr(Settings, "_llm", None) is None:
                    raise RuntimeError(
                        "LLM not properly configured in unified settings. "
                        "Please ensure LlamaIndex Settings are initialized."
                    )
                logger.info("LlamaIndex LLM initialized from Settings")

                # Build expensive components as locals. Publication shares the
                # lifecycle lock with close, so closed coordinators cannot reopen.
                model = build_chat_model(
                    settings,
                    timeout_cap=self.max_agent_timeout,
                )
                logger.info("LangChain chat model initialized from unified settings")
                self._ensure_checkpointer()
                checkpointer = self.checkpointer
                if checkpointer is None:
                    raise RuntimeError("LangGraph checkpointer is not initialized")
                agents, graph, compiled_graph = self._build_agent_graph_components(
                    model=model,
                    checkpointer=checkpointer,
                )

                with self._graph_runner_lock:
                    if self._closed:
                        return False
                    self.llm = model
                    self.agents = agents
                    self.graph = graph
                    self.compiled_graph = compiled_graph
                    self._setup_complete = True
                logger.info("Agent graph setup completed successfully")
                return True

            except Exception as e:
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
        if self.llm is None:
            raise RuntimeError("LangChain chat model is not initialized")
        agents, graph, compiled_graph = self._build_agent_graph_components(
            model=self.llm,
            checkpointer=self.checkpointer,
        )
        self.agents = agents
        self.graph = graph
        self.compiled_graph = compiled_graph
        logger.info("Agent graph setup completed successfully")

    def _build_agent_graph_components(
        self,
        *,
        model: Any,
        checkpointer: Any,
    ) -> tuple[dict[str, Any], Any, Any]:
        """Build graph components without publishing partial coordinator state."""
        try:
            agent_tool_sets = build_agent_tool_sets(settings)

            hook_middleware = _AgentHookMiddleware(
                pre_model_hook=self._create_pre_model_hook(),
                post_model_hook=self._create_post_model_hook(),
            )

            agents: dict[str, Any] = {}
            for name, tools in agent_tool_sets.items():
                agents[name] = create_agent(
                    model,
                    tools=tools,
                    state_schema=MultiAgentGraphState,
                    context_schema=AgentRuntimeContext,
                    middleware=[hook_middleware],
                    name=name,
                    store=self.store,
                )

            # Create supervisor system prompt
            system_prompt = self._create_supervisor_prompt()

            # Create list of agents for supervisor preserving definition order.
            supervisor_agents = list(agents.values())

            # Graph-native supervisor (ADR-011): supervisor + subagents via
            # `StateGraph`, using one atomic dispatch tool that returns
            # `Command.PARENT`.
            graph = build_multi_agent_supervisor_graph(
                supervisor_agents,
                model=model,
                prompt=system_prompt,
                state_schema=MultiAgentGraphState,
                context_schema=AgentRuntimeContext,
                middleware=[hook_middleware],
                params=SupervisorBuildParams(
                    supervisor_name="supervisor",
                    output_mode="last_message",
                    add_handoff_messages=True,
                    add_handoff_back_messages=True,
                ),
            )

            # Compile graph with checkpointer + optional store (ADR-058).
            compiled_graph = graph.compile(
                checkpointer=checkpointer,
                store=self.store,
            )

            return agents, graph, compiled_graph

        except Exception as e:
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
            "- planner_agent: Complex query decomposition\n"
            "- retrieval_agent: Document search through the native retrieval router\n"
            "- synthesis_agent: Multi-source result combination\n"
            "- validation_agent: Response quality validation\n\n"
            "Coordination strategy:\n"
            "1. Use planner_agent only when decomposition is necessary\n"
            "2. Execute retrieval_agent; its RouterQueryEngine selects the strategy\n"
            "3. Use synthesis_agent for multi-source results\n"
            "4. End with validation_agent for quality assurance\n\n"
            "Call dispatch_agents at most once per coordination step. Put every "
            "independent worker in that call's unique destinations list so they run "
            "in parallel. Minimize unnecessary worker calls. When the task is "
            "complete, answer directly without another dispatch."
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
            # Normalize flag naming across code paths
            "parallel_execution_active": bool(
                state.get("parallel_execution_active")
                or state.get("parallel_tool_calls")
            ),
            "optimization_enabled": True,
            "model_path": settings.effective_model,
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
                "context_window_used": settings.effective_context_window,
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
    ) -> dict[str, Any]:
        """Constructs the starting state dictionary for the agent graph.

        Args:
            query: The processed user input string.
            start_time: Wall-clock start timestamp for the request.

        Returns:
            A dictionary conforming to MultiAgentState schema.
        """
        return MultiAgentState(
            messages=[HumanMessage(content=query)],
            total_start_time=start_time,
            output_mode="last_message",
            parallel_execution_active=True,
            deadline_ts=time.monotonic() + self.max_agent_timeout,
        ).model_dump()

    def _handle_timeout_response(
        self,
        start_time: float,
        *,
        cancel_reason: str,
    ) -> AgentResponse:
        """Orchestrates the response when the agent workflow exceeds its timeout.

        Args:
            start_time: Wall-clock start time for the entire request.
            cancel_reason: Stable timeout classification.

        Returns:
            A standardized timeout response.
        """
        processing_time = time.perf_counter() - start_time
        return AgentResponse(
            content=("The multi-agent system timed out while processing your request."),
            sources=[],
            metadata={"reason": "timeout", "cancel_reason": cancel_reason},
            validation_score=0.0,
            processing_time=processing_time,
            optimization_metrics={"timeout": True},
        )

    def _handle_stopped_response(
        self,
        start_time: float,
        *,
        cancel_reason: str,
    ) -> AgentResponse:
        """Return the canonical response for a non-timeout workflow stop."""
        return AgentResponse(
            content="The multi-agent system stopped before completing your request.",
            sources=[],
            metadata={
                "reason": "workflow_stopped",
                "cancel_reason": cancel_reason,
            },
            validation_score=0.0,
            processing_time=time.perf_counter() - start_time,
            optimization_metrics={"stopped": True},
        )

    def _handle_workflow_result(
        self,
        result: dict[str, Any] | None,
        query: str,
        start_time: float,
        coordination_time: float,
    ) -> tuple[AgentResponse, bool, bool]:
        """Analyzes the agent workflow output and resolves it to a final response.

        Args:
            result: Terminal graph state or None on critical failure.
            query: Original query string.
            start_time: Processing start timestamp.
            coordination_time: Total accumulated overhead in coordination.

        Returns:
            A tuple of response, stopped flag, and timed-out flag.
        """
        workflow_stopped = bool(
            isinstance(result, dict) and result.get("workflow_stopped")
        )
        workflow_timed_out = bool(
            workflow_stopped and isinstance(result, dict) and result.get("timed_out")
        )
        if not workflow_stopped:
            response = self._extract_response(
                result or {}, query, start_time, coordination_time
            )
            return response, False, False

        cancel_reason = str((result or {}).get("cancel_reason") or "workflow_cancelled")
        if workflow_timed_out:
            logger.warning("Coordinator detected timeout ({})", cancel_reason)
            response = self._handle_timeout_response(
                start_time,
                cancel_reason=cancel_reason,
            )
        else:
            logger.warning("Coordinator detected workflow stop ({})", cancel_reason)
            response = self._handle_stopped_response(
                start_time,
                cancel_reason=cancel_reason,
            )
        self._record_query_metrics(time.perf_counter() - start_time, False)
        return response, True, workflow_timed_out

    def _annotate_span(
        self,
        span: Span,
        workflow_stopped: bool,
        workflow_timed_out: bool,
        processing_time: float,
    ) -> None:
        """Attaches final outcome attributes to the OpenTelemetry trace span.

        Args:
            span: The span to annotate.
            workflow_stopped: Whether execution stopped without a terminal result.
            workflow_timed_out: Final timeout status.
            processing_time: Terminal duration in seconds.
        """
        span.set_attribute("coordinator.workflow_stopped", bool(workflow_stopped))
        span.set_attribute("coordinator.workflow_timeout", bool(workflow_timed_out))
        span.set_attribute("coordinator.success", not workflow_stopped)
        span.set_attribute(
            "coordinator.processing_time_ms", round(processing_time * 1000.0, 3)
        )

    def process_query(
        self,
        query: str,
        *,
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
            settings_override: Dictionary of configuration overrides.
            thread_id: Identifier for cross-request conversation state.
            user_id: Namespace for memory and policy scoping.
            checkpoint_id: Optional ID to resume from a specific state.

        Returns:
            An AgentResponse containing content, sources, and metrics.
        """
        start_time = time.perf_counter()

        # Ensure setup is complete
        if not self._ensure_setup():
            return self._create_error_response("initialization_failed", start_time)

        with contextlib.ExitStack() as exit_stack:
            span = self._start_span(exit_stack, thread_id, len(query))

            try:
                # Initialize state with execution parameters
                initial_state = self._build_initial_state(query, start_time)

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

                response, workflow_stopped, workflow_timed_out = (
                    self._handle_workflow_result(
                        result, query, start_time, coordination_time
                    )
                )

                processing_time = time.perf_counter() - start_time

                # Memory consolidation (SPEC-041)
                if not workflow_stopped:
                    self._schedule_memory_consolidation(
                        result or {}, thread_id=thread_id, user_id=user_id
                    )

                # Best-effort analytics logging (never impact user flow)
                if not workflow_stopped:
                    self._record_query_metrics(processing_time, True)

                # Validate performance targets
                if coordination_time > COORDINATION_OVERHEAD_THRESHOLD:
                    logger.warning(
                        "Coordination overhead {overhead:.3f}s exceeds threshold",
                        overhead=coordination_time,
                    )

                self._annotate_span(
                    span,
                    workflow_stopped,
                    workflow_timed_out,
                    processing_time,
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

                return self._create_error_response("execution_failed", start_time)

    def _run_agent_workflow(  # noqa: PLR0911, PLR0915 - explicit terminal states
        self,
        initial_state: dict[str, Any],
        *,
        thread_id: str,
        user_id: str,
        checkpoint_id: str | None,
        runtime_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Execute the graph asynchronously behind a strict synchronous deadline.

        Args:
            initial_state: The starting dictionary for graph execution.
            thread_id: Persistence thread identifier.
            user_id: Persistence namespace for memory.
            checkpoint_id: Optional ID to resume from a prior state.
            runtime_context: Key-value pairs injected into tool execution.

        Returns:
            The terminal state dictionary, including canonical stop metadata.
        """
        try:
            if self.compiled_graph is None:
                raise RuntimeError("Agent graph is not compiled")

            now = time.monotonic()
            deadline_ts = min(
                _require_deadline_ts(initial_state),
                now + self.max_agent_timeout,
            )
            remaining = max(0.0, deadline_ts - now)
            if remaining <= 0:
                return self._build_workflow_stopped_state(
                    initial_state,
                    cancel_reason="deadline_exceeded",
                )

            config = _checkpoint_config(
                thread_id=thread_id,
                user_id=user_id,
                checkpoint_id=checkpoint_id,
            )
            initial_state["memory_generations"] = capture_memory_namespace_generations(
                user_id=user_id,
                thread_id=thread_id,
            )
            configurable = config.get("configurable")
            if not isinstance(configurable, dict):
                raise RuntimeError("Checkpoint configuration is invalid")
            persistence_id = str(configurable["thread_id"])
            active_run = self._begin_active_run(persistence_id)
            if active_run is None:
                with self._active_runs_lock:
                    run_is_purged = persistence_id in self._purged_persistence_ids
                return self._build_workflow_stopped_state(
                    initial_state,
                    cancel_reason=(
                        "session_purged" if run_is_purged else "previous_run_active"
                    ),
                )

            try:
                graph = cast(Any, self.compiled_graph).copy({"step_timeout": remaining})
            except BaseException:
                self._finish_active_run(persistence_id, active_run)
                raise
            coroutine = self._consume_agent_workflow(
                graph,
                initial_state,
                config=config,
                runtime_context=runtime_context,
                deadline_ts=deadline_ts,
                persistence_id=persistence_id,
                public_thread_id=str(thread_id),
                active_run=active_run,
            )
            try:
                future = self._get_graph_runner().submit(coroutine)
            except _GraphRunnerCapacityError:
                self._finish_active_run(persistence_id, active_run)
                return self._build_workflow_stopped_state(
                    initial_state,
                    cancel_reason="runner_saturated",
                )
            except RuntimeError:
                coroutine.close()
                self._finish_active_run(persistence_id, active_run)
                if self._closed:
                    return self._build_workflow_stopped_state(
                        initial_state,
                        cancel_reason="coordinator_closed",
                    )
                raise
            except BaseException:
                coroutine.close()
                self._finish_active_run(persistence_id, active_run)
                raise

            try:
                return future.result(timeout=max(0.0, deadline_ts - time.monotonic()))
            except FuturesTimeoutError:
                deadline_reached = not future.done() or time.monotonic() >= deadline_ts
                cancel_reason = (
                    "deadline_exceeded" if deadline_reached else "dependency_timeout"
                )
                active_run.control.request_drain(cancel_reason)
                future.cancel()
                return self._build_workflow_stopped_state(
                    initial_state,
                    cancel_reason=cancel_reason,
                )
            except (FuturesCancelledError, GraphDrained):
                cancel_reason = active_run.control.drain_reason or "workflow_cancelled"
                future.cancel()
                return self._build_workflow_stopped_state(
                    initial_state,
                    cancel_reason=cancel_reason,
                )

        except (RuntimeError, ValueError, AttributeError, TimeoutError) as e:
            redaction = build_pii_log_entry(str(e), key_id="agent_workflow.exception")
            logger.error(
                "Agent workflow execution failed (error_type={}, error={})",
                type(e).__name__,
                redaction.redacted,
            )
            raise

    async def _consume_agent_workflow(
        self,
        graph: Any,
        initial_state: dict[str, Any],
        *,
        config: RunnableConfig,
        runtime_context: dict[str, Any] | None,
        deadline_ts: float,
        persistence_id: str,
        public_thread_id: str,
        active_run: _ActiveAgentRun,
    ) -> dict[str, Any]:
        """Consume one async graph stream and release its thread fence on exit."""
        result: dict[str, Any] | None = None
        try:
            remaining = max(0.0, deadline_ts - time.monotonic())
            async with asyncio.timeout(remaining):
                async for state in graph.astream(
                    initial_state,
                    config=config,
                    context=runtime_context,
                    stream_mode="values",
                    control=active_run.control,
                ):
                    result = _as_state_dict(state)

            cancel_reason = active_run.control.drain_reason
            if cancel_reason is None and time.monotonic() >= deadline_ts:
                cancel_reason = "deadline_exceeded"
                active_run.control.request_drain(cancel_reason)
            if cancel_reason is not None:
                return self._build_workflow_stopped_state(
                    initial_state,
                    cancel_reason=cancel_reason,
                )
            if result is not None:
                # Capture provenance before releasing the per-thread run fence.
                # Remove an input checkpoint selector so this resolves the new
                # terminal head rather than the historical source checkpoint.
                latest_configurable = dict(config.get("configurable") or {})
                latest_configurable.pop("checkpoint_id", None)
                try:
                    terminal_state = await graph.aget_state(
                        {"configurable": latest_configurable}
                    )
                    terminal_config = getattr(terminal_state, "config", None)
                    terminal_values = (
                        terminal_config.get("configurable")
                        if isinstance(terminal_config, dict)
                        else None
                    )
                    terminal_checkpoint_id = (
                        terminal_values.get("checkpoint_id")
                        if isinstance(terminal_values, dict)
                        else None
                    )
                    if terminal_checkpoint_id:
                        result["_terminal_checkpoint_id"] = str(terminal_checkpoint_id)
                except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
                    logger.debug(
                        "Terminal checkpoint provenance unavailable (error_type={})",
                        type(exc).__name__,
                    )
                return result
            thread_redacted = build_pii_log_entry(
                public_thread_id, key_id="coordinator.thread_id"
            ).redacted
            logger.warning(
                "Agent workflow produced no result; classifying run as "
                "stopped (thread_id={})",
                thread_redacted,
            )
            return self._build_workflow_stopped_state(
                initial_state,
                cancel_reason="workflow_no_result",
            )
        finally:
            self._finish_active_run(persistence_id, active_run)

    def _build_workflow_stopped_state(
        self,
        initial_state: dict[str, Any],
        *,
        cancel_reason: str,
    ) -> dict[str, Any]:
        """Build and record the canonical stopped graph state."""
        result = dict(initial_state)
        try:
            started_at = float(initial_state.get("total_start_time", 0.0))
        except (TypeError, ValueError):
            started_at = 0.0
        elapsed = max(0.0, time.perf_counter() - started_at)
        timed_out = cancel_reason in _TIMEOUT_REASONS
        result["workflow_stopped"] = True
        result["timed_out"] = timed_out
        result["deadline_s"] = self.max_agent_timeout
        result["cancel_reason"] = cancel_reason
        logger.warning(
            "Agent workflow stopped after {:.2f}s ({})",
            elapsed,
            cancel_reason,
        )
        event: dict[str, Any] = {
            "agent_workflow_stopped": True,
            "decision_timeout_s": self.max_agent_timeout,
            "elapsed_s": elapsed,
            "cancel_reason": cancel_reason,
        }
        if cancel_reason == "deadline_exceeded":
            event["agent_deadline_exceeded"] = True
        with contextlib.suppress(Exception):  # pragma: no cover - telemetry
            log_jsonl(event)
        return result

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
            sources: list[dict[str, Any]] = []
            turn_id_value = final_state.get("turn_id")
            turn_id = turn_id_value.strip() if isinstance(turn_id_value, str) else ""
            synthesis_result = final_state.get("synthesis_result", {})
            current_batches = current_retrieval_batches(final_state)
            current_watermark = (
                retrieval_batch_watermark(current_batches)
                if current_batches is not None
                else None
            )
            if (
                isinstance(synthesis_result, dict)
                and synthesis_result.get("turn_id") == turn_id
                and current_watermark is not None
                and synthesis_result.get("retrieval_watermark") == current_watermark
                and isinstance(synthesis_result.get("documents"), list)
            ):
                sources = [
                    source
                    for source in synthesis_result["documents"]
                    if isinstance(source, dict)
                ]
            else:
                if current_batches is not None:
                    sources = interleave_retrieval_documents(current_batches)[
                        :MAX_RETRIEVAL_RESULTS
                    ]

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
                "planning_output": final_state.get("planning_output", {}),
                "agent_timings": final_state.get("agent_timings", {}),
                "validation_result": validation_result,
                "processing_time_ms": round(processing_time * 1000, 2),
                "agents_used": list(final_state.get("agent_timings", {}).keys()),
                "errors": final_state.get("errors", []),
                "system_info": {
                    "agents_used": 4,
                    "model": settings.effective_model,
                    "framework": "LangGraph StateGraph Supervisor",
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
        self,
        reason: Literal["initialization_failed", "execution_failed"],
        start_time: float,
    ) -> AgentResponse:
        """Generates a standardized error response with timing metadata.

        Args:
            reason: Stable, non-sensitive failure reason.
            start_time: Processing start timestamp.

        Returns:
            An AgentResponse indicating initialization or execution failure.
        """
        processing_time = time.perf_counter() - start_time
        content = (
            "Unable to initialize the coordinator."
            if reason == "initialization_failed"
            else "Unable to process the query."
        )
        return AgentResponse(
            content=content,
            sources=[],
            metadata={"reason": reason},
            validation_score=0.0,
            processing_time=processing_time,
            optimization_metrics={"error": True},
        )

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
        config = _checkpoint_config(
            thread_id=thread_id,
            user_id=user_id,
            checkpoint_id=checkpoint_id,
        )
        with self._persistence_lock:
            if self._closed:
                return {}
            snap = self.compiled_graph.get_state(config)
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
        config = _checkpoint_config(thread_id=thread_id, user_id=user_id)
        out: list[dict[str, Any]] = []
        try:
            with self._persistence_lock:
                if self._closed:
                    return []
                for snap in self.compiled_graph.get_state_history(
                    config, limit=int(limit)
                ):
                    snapshot_config = getattr(snap, "config", None)
                    conf = (
                        snapshot_config.get("configurable")
                        if isinstance(snapshot_config, dict)
                        else None
                    )
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

    def fork_from_checkpoint(  # noqa: PLR0911 - fail-closed persistence boundary
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

        config = _checkpoint_config(
            thread_id=thread_id,
            user_id=user_id,
            checkpoint_id=checkpoint_id,
        )
        configurable = config.get("configurable")
        if not isinstance(configurable, dict):
            return None
        persistence_id = str(configurable.get("thread_id") or "")
        thread_redacted = build_pii_log_entry(
            str(thread_id), key_id="coordinator.thread_id"
        ).redacted
        checkpoint_redacted = build_pii_log_entry(
            str(checkpoint_id), key_id="coordinator.checkpoint_id"
        ).redacted

        # Fetch and fork under one mutation lock while occupying the graph-run
        # fence. A new turn cannot begin between the active-run check and the
        # branch write; purge either deletes a completed fork or fails it closed.
        fork_run: _ActiveAgentRun | None = None
        try:
            with self._persistence_lock:
                if self._closed:
                    return None
                fork_run = self._begin_active_run(persistence_id)
                if fork_run is None:
                    return None
                source_state = self.compiled_graph.get_state(config)
                if not source_state or not source_state.values:
                    logger.debug(
                        "Checkpoint not found or empty (thread_id={} checkpoint_id={})",
                        thread_redacted,
                        checkpoint_redacted,
                    )
                    return None
                # ``__copy__`` clones the selected checkpoint into a new head
                # without running the workflow.
                new_config = self.compiled_graph.update_state(
                    config, None, as_node="__copy__"
                )
        except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
            err = build_pii_log_entry(str(exc), key_id="coordinator.checkpoint_fork")
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
        finally:
            if fork_run is not None:
                self._finish_active_run(persistence_id, fork_run)

        conf = new_config.get("configurable") if isinstance(new_config, dict) else None
        conf = conf if isinstance(conf, dict) else {}
        new_checkpoint_id = conf.get("checkpoint_id")
        return str(new_checkpoint_id) if new_checkpoint_id else None

    @staticmethod
    def _remaining_purge_budget(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    @classmethod
    def _acquire_before(cls, lock: Any, deadline: float) -> bool:
        remaining = cls._remaining_purge_budget(deadline)
        return remaining > 0 and lock.acquire(timeout=remaining)

    def _register_session_purge(
        self,
        *,
        namespace: tuple[str, ...],
        persistence_id: str,
        deadline: float,
    ) -> tuple[bool, _ActiveAgentRun | None]:
        if not self._acquire_before(self._persistence_lock, deadline):
            return False, None
        try:
            if self._closed or not self._acquire_before(
                self._active_runs_lock, deadline
            ):
                return False, None
            try:
                remaining = self._remaining_purge_budget(deadline)
                if (
                    try_tombstone_memory_namespace(namespace, timeout_s=remaining)
                    is None
                ):
                    return False, None
                self._purged_persistence_ids.add(persistence_id)
                return True, self._active_runs.get(persistence_id)
            finally:
                self._active_runs_lock.release()
        finally:
            self._persistence_lock.release()

    @classmethod
    def _delete_session_within_budget(
        cls,
        conn: sqlite3.Connection,
        *,
        thread_id: str,
        user_id: str,
        deadline: float,
    ) -> bool:
        if cls._remaining_purge_budget(deadline) <= 0:
            return False
        previous_busy_timeout: int | None = None
        try:
            if isinstance(conn, sqlite3.Connection):
                row = conn.execute("PRAGMA busy_timeout;").fetchone()
                previous_busy_timeout = int(row[0]) if row is not None else None
                remaining_ms = max(
                    1,
                    math.ceil(cls._remaining_purge_budget(deadline) * 1000),
                )
                conn.execute(f"PRAGMA busy_timeout={remaining_ms};")
            delete_persisted_session(
                conn,
                thread_id=thread_id,
                user_id=user_id,
            )
        except Exception as exc:
            redaction = build_pii_log_entry(
                str(exc), key_id="coordinator.session_purge"
            )
            logger.warning(
                "Session purge failed (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            return False
        finally:
            if previous_busy_timeout is not None:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute(f"PRAGMA busy_timeout={previous_busy_timeout};")
        return True

    def _delete_fenced_session(
        self,
        conn: sqlite3.Connection,
        *,
        thread_id: str,
        user_id: str,
        persistence_id: str,
        deadline: float,
    ) -> bool:
        if not self._acquire_before(self._persistence_lock, deadline):
            return False
        try:
            if self._closed or not self._acquire_before(
                self._active_runs_lock, deadline
            ):
                return False
            try:
                if persistence_id in self._active_runs:
                    return False
            finally:
                self._active_runs_lock.release()
            return self._delete_session_within_budget(
                conn,
                thread_id=thread_id,
                user_id=user_id,
                deadline=deadline,
            )
        finally:
            self._persistence_lock.release()

    def purge_session(
        self,
        *,
        conn: sqlite3.Connection,
        thread_id: str,
        user_id: str,
        timeout_s: float = SESSION_PURGE_DRAIN_TIMEOUT_S,
    ) -> bool:
        """Fence, drain, and durably delete one session within one time budget."""
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout < 0:
            return False
        deadline = time.monotonic() + timeout
        namespace = memory_namespace(user_id=user_id, thread_id=thread_id)
        persistence_id = checkpoint_thread_id(thread_id=thread_id, user_id=user_id)
        registered, active_run = self._register_session_purge(
            namespace=namespace,
            persistence_id=persistence_id,
            deadline=deadline,
        )
        if not registered:
            return False
        if active_run is not None:
            active_run.control.request_drain("session_purged")
            if not active_run.finished.wait(
                timeout=self._remaining_purge_budget(deadline)
            ):
                logger.warning("Session purge deferred: active graph run did not drain")
                return False
        return self._delete_fenced_session(
            conn,
            thread_id=thread_id,
            user_id=user_id,
            persistence_id=persistence_id,
            deadline=deadline,
        )

    def _consolidate_memories(
        self,
        final_state: dict[str, Any],
        thread_id: str,
        user_id: str,
        expected_generation: int,
        checkpoint_id: str,
    ) -> None:
        """Executes background memory consolidation for a completed turn.

        Extracts potential memory candidates from history and merges them into
        the long-term store based on configured similarity and importance
        policies.

        Args:
            final_state: The terminal state of the agent graph.
            thread_id: Unique conversation identifier.
            user_id: User namespace for memory scoping.
            expected_generation: Namespace generation captured before scheduling.
            checkpoint_id: Terminal checkpoint that produced ``final_state``.
        """
        logger.debug("Starting background memory consolidation")
        namespace = memory_namespace(user_id=user_id, thread_id=thread_id)
        if (
            self.store is None
            or is_memory_namespace_tombstoned(namespace)
            or memory_namespace_generation(namespace) != expected_generation
        ):
            return

        try:
            deadline_ts = time.monotonic() + MEMORY_CONSOLIDATION_TIMEOUT_S

            def _timed_out() -> bool:
                return time.monotonic() >= deadline_ts

            if _timed_out():
                logger.debug("Memory consolidation timed out before start")
                return

            policy = MemoryConsolidationPolicy.from_settings()

            # 2. Extract candidates from the conversation turn
            messages = final_state.get("messages", [])
            candidates = extract_memory_candidates(
                messages,
                checkpoint_id=checkpoint_id,
                llm=self.llm,
                policy=policy,
                deadline_ts=deadline_ts,
            )
            if not candidates:
                return
            if _timed_out():
                logger.debug("Memory consolidation timed out")
                return

            with memory_namespace_lock(namespace):
                if (
                    self._closed
                    or is_memory_namespace_tombstoned(namespace)
                    or memory_namespace_generation(namespace) != expected_generation
                ):
                    return
                consolidate_and_apply_memory_candidates(
                    candidates,
                    self.store,
                    namespace,
                    deadline_ts=deadline_ts,
                    policy=policy,
                    expected_generation=expected_generation,
                )

        except Exception as exc:
            redaction = build_pii_log_entry(
                str(exc), key_id="coordinator.memory_consolidation_background"
            )
            logger.debug(
                "Memory consolidation background task failed (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )

    def _schedule_memory_consolidation(  # noqa: PLR0915 - bounded lifecycle cleanup
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
        namespace = memory_namespace(user_id=user_id, thread_id=thread_id)
        checkpoint_id_value = final_state.get("_terminal_checkpoint_id")
        if not isinstance(checkpoint_id_value, str) or not checkpoint_id_value:
            logger.debug(
                "Skipping memory consolidation; terminal checkpoint provenance missing"
            )
            return
        checkpoint_id = checkpoint_id_value
        expected_generation = memory_generation_from_state(final_state, "session")
        if expected_generation is None:
            logger.debug(
                "Skipping memory consolidation; admitted generation provenance missing"
            )
            return
        with memory_namespace_lock(namespace):
            if (
                self.store is None
                or self._closed
                or self._memory_executor_closed
                or is_memory_namespace_tombstoned(namespace)
                or memory_namespace_generation(namespace) != expected_generation
            ):
                return

        with self._memory_jobs_lock:
            if self._closed or self._memory_executor_closed:
                return
            self._memory_jobs[namespace] = self._memory_jobs.get(namespace, 0) + 1

        def _forget_job() -> None:
            with self._memory_jobs_lock:
                remaining = self._memory_jobs.get(namespace, 0) - 1
                if remaining > 0:
                    self._memory_jobs[namespace] = remaining
                else:
                    self._memory_jobs.pop(namespace, None)

        if not self._memory_consolidation_semaphore.acquire(blocking=False):
            _forget_job()
            logger.debug("Skipping memory consolidation; max in-flight reached")
            return

        def _release_slot() -> None:
            with contextlib.suppress(ValueError):
                self._memory_consolidation_semaphore.release()

        try:
            future = self._memory_executor.submit(
                self._consolidate_memories,
                final_state,
                thread_id,
                user_id,
                expected_generation,
                checkpoint_id,
            )
        except RuntimeError as exc:
            _release_slot()
            _forget_job()
            logger.debug(
                "Memory consolidation submit skipped (error_type={})",
                type(exc).__name__,
            )
            return

        future.add_done_callback(lambda _completed: _forget_job())

        def _watch_future() -> None:
            should_release = True
            released = threading.Event()

            def _release_once(reason: str) -> None:
                if released.is_set():
                    return
                released.set()
                logger.debug(
                    "Releasing memory consolidation slot ({})",
                    reason,
                )
                _release_slot()

            try:
                future.result(timeout=MEMORY_CONSOLIDATION_TIMEOUT_S)
            except FuturesTimeoutError:
                logger.warning(
                    "Memory consolidation timed out after {}s",
                    MEMORY_CONSOLIDATION_TIMEOUT_S,
                )
                if not future.cancel():
                    logger.debug("Memory consolidation still running after timeout")

                    # Attach callback to release slot when future eventually completes.
                    # This prevents semaphore leak when task doesn't respond to cancel.
                    def _on_future_done(fut: Any) -> None:
                        with contextlib.suppress(Exception):
                            fut.result()  # Log any exception but don't propagate
                        _release_once("done")

                    try:
                        future.add_done_callback(_on_future_done)
                        should_release = False
                    except (RuntimeError, TypeError) as exc:
                        redaction = build_pii_log_entry(
                            str(exc),
                            key_id="coordinator.memory_consolidation_add_callback",
                        )
                        logger.debug(
                            "Could not attach memory completion callback "
                            "(error_type={} error={}); waiting for worker exit",
                            type(exc).__name__,
                            redaction.redacted,
                        )
                        with contextlib.suppress(Exception):
                            future.result()
                        _release_once("callback-failed-done")
                        should_release = False
            except Exception as exc:
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
