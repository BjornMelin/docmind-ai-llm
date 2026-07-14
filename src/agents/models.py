"""Agent-specific Pydantic models for DocMind AI Multi-Agent System.

This module contains data models specifically designed for the multi-agent coordination
system, including state management and response schemas. These models are closely
coupled to LangGraph patterns and agent coordination logic.

HYBRID MODEL ORGANIZATION:
Following the hybrid model organization strategy, these models are colocated within
the agents domain because they are:
- Tightly coupled to agent coordination workflows
- Specific to LangGraph supervisor patterns
- Not used outside the agent coordination system
- Domain-specific to multi-agent orchestration

Models:
    AgentResponse: Response model from multi-agent coordination system
    MultiAgentState: Extended state for LangGraph multi-agent coordination

Architecture Decision:
    These models are placed within the agents module following LangGraph best practices
    and domain-driven design principles. They are domain-specific and tightly coupled
    to the agent coordination system, making this the most appropriate location per
    the hybrid organization strategy.
"""

import uuid
from typing import Annotated, Any, NotRequired

from langchain.agents import AgentState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

MAX_RETRIEVAL_HISTORY_BATCHES = 32


def merge_retrieval_results(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Append retrieval batches while bounding checkpoint growth.

    Taking the tail after each merge is associative, so this reducer remains safe
    when LangGraph combines parallel worker updates in either order.
    """
    return [*left, *right][-MAX_RETRIEVAL_HISTORY_BATCHES:]


class AgentRuntimeContext(TypedDict):
    """Transient, non-checkpointed dependencies available to agent tools."""

    router_engine: NotRequired[Any]


class MemoryNamespaceGenerations(TypedDict):
    """Memory mutation generations captured when a graph turn is admitted."""

    session: int
    user: int


class AgentResponse(BaseModel):
    """Response from ADR-compliant multi-agent coordination system.

    This model represents the structured output from the multi-agent system,
    including content, sources, metadata, and performance metrics specific
    to the agent coordination workflow.

    Attributes:
        content: Generated response content from the multi-agent system
        sources: List of source documents used in generation
        metadata: Agent processing metadata and coordination information
        validation_score: Response validation confidence score (0.0-1.0)
        processing_time: Total processing time in seconds
        optimization_metrics: FP8 optimization and parallel execution metrics
        agent_decisions: Decisions made by agents during processing
    """

    content: str = Field(description="Generated response content")
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="Source documents"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Agent processing metadata"
    )
    validation_score: float = Field(
        default=0.0, description="Response validation confidence", ge=0.0, le=1.0
    )
    processing_time: float = Field(
        default=0.0, description="Total processing time in seconds"
    )
    optimization_metrics: dict[str, Any] = Field(
        default_factory=dict, description="FP8 and parallel execution metrics"
    )
    agent_decisions: list[dict[str, Any]] = Field(
        default_factory=list, description="Decisions made by agents during processing"
    )


class MultiAgentGraphState(AgentState[Any]):
    """TypedDict state schema for LangChain/LangGraph agent graphs.

    ``langchain.agents.create_agent`` requires a ``TypedDict`` that extends
    ``AgentState``. The Pydantic ``MultiAgentState`` model remains the local
    validation/serialization helper used by DocMind code and tests.
    """

    planning_output: NotRequired[dict[str, Any]]
    retrieval_results: NotRequired[
        Annotated[list[dict[str, Any]], merge_retrieval_results]
    ]
    turn_id: str
    synthesis_result: NotRequired[dict[str, Any]]
    validation_result: NotRequired[dict[str, Any]]
    agent_timings: NotRequired[dict[str, float]]
    total_start_time: NotRequired[float]
    parallel_execution_active: NotRequired[bool]
    token_reduction_achieved: NotRequired[float]
    context_trimmed: NotRequired[bool]
    tokens_trimmed: NotRequired[int]
    output_mode: NotRequired[str]
    errors: NotRequired[list[str]]
    remaining_steps: NotRequired[int]
    workflow_stopped: NotRequired[bool]
    timed_out: NotRequired[bool]
    deadline_s: NotRequired[float]
    deadline_ts: float
    memory_generations: MemoryNamespaceGenerations
    cancel_reason: NotRequired[str]


class MultiAgentState(BaseModel):
    """Extended state for ADR-compliant multi-agent coordination.

    This model extends LangGraph's MessagesState to provide comprehensive state
    management for the multi-agent system, including agent decisions, performance
    tracking, context management, and error handling.

    Inherits:
        MessagesState: Base LangGraph state with message handling capabilities

    Additional Fields:
        context: Chat memory buffer for conversation context
        planning_output: Output from planning agent
        retrieval_results: Results from retrieval agent
        synthesis_result: Result from synthesis agent
        validation_result: Result from validation agent
        agent_timings: Performance timing for each agent
        total_start_time: Start time of the coordination workflow
        parallel_execution_active: Whether parallel execution is active
        token_reduction_achieved: Token reduction percentage achieved
        context_trimmed: Whether context was trimmed for token limits
        tokens_trimmed: Number of tokens trimmed
        output_mode: Output formatting mode
        errors: List of errors encountered during processing
        remaining_steps: Remaining steps for LangGraph supervisor
        workflow_stopped: Whether the workflow stopped before a terminal result
        timed_out: Whether the workflow timed out
        deadline_s: Decision timeout budget in seconds
        deadline_ts: Absolute deadline timestamp (time.monotonic)
        cancel_reason: Cancellation reason string, when applicable
    """

    # Core state
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    # Agent decisions and results
    planning_output: dict[str, Any] = Field(default_factory=dict)
    retrieval_results: list[dict[str, Any]] = Field(default_factory=list)
    turn_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    synthesis_result: dict[str, Any] = Field(default_factory=dict)
    validation_result: dict[str, Any] = Field(default_factory=dict)

    # Performance tracking (ADR-011)
    agent_timings: dict[str, float] = Field(default_factory=dict)
    total_start_time: float = Field(default=0.0)
    parallel_execution_active: bool = Field(default=False)
    token_reduction_achieved: float = Field(default=0.0)

    # Context management (ADR-004, ADR-011)
    context_trimmed: bool = Field(default=False)
    tokens_trimmed: int = Field(default=0)

    # Output mode configuration (ADR-011)
    output_mode: str = Field(default="structured")

    # Error handling
    errors: list[str] = Field(default_factory=list)

    # LangGraph supervisor requirements (ADR-011)
    remaining_steps: int = Field(
        default=10, description="Remaining steps for supervisor"
    )
    workflow_stopped: bool = Field(default=False)
    timed_out: bool = Field(default=False)
    deadline_s: float = Field(default=0.0)
    deadline_ts: float
    memory_generations: MemoryNamespaceGenerations | None = None
    cancel_reason: str | None = Field(default=None)
