"""Data schemas for DocMind AI with Multi-Agent Coordination.

This module contains shared Pydantic models used across multiple modules in the
DocMind AI system. Following the hybrid model organization strategy, this module
centralizes models that are:

- Used by multiple modules throughout the codebase
- Part of the public API interface
- Core data structures shared across domains

HYBRID MODEL ORGANIZATION STRATEGY:

1. **Centralized Shared Models** (this module):
   - Document: Core document representation used throughout the system
   - QueryRequest: API request model used by multiple components
   - AgentDecision: Decision tracking used across agent coordination
   - ConversationTurn/ConversationContext: Chat memory shared across agents
   - PerformanceMetrics: System monitoring shared across components
   - ValidationResult: Response validation shared across agents
   - ErrorResponse: Error handling shared across the system
   - AnalysisOutput: Structured output format for LLM responses

2. **Domain-Specific Models** (colocated with usage):
   - Agent models: src/agents/models.py (AgentResponse, MultiAgentState)
   - Retrieval models: src/retrieval/*/*.py (ClipConfig, PropertyGraphConfig,
     DSPyConfig)
   - Configuration models: src/config/*.py (VLLMConfig)
   - Feature-specific models: Kept with their respective modules

3. **Benefits**:
   - Reduces circular imports by centralizing shared models
   - Maintains domain cohesion for specialized models
   - Simplifies maintenance and testing
   - Clear separation of concerns

4. **Import Conventions**:
   - Shared models: `from src.models.schemas import Document, QueryRequest`
   - Agent models: `from src.agents.models import AgentResponse, MultiAgentState`
   - Domain models: Import from their colocated modules

This organization follows Domain-Driven Design principles while maintaining
practical import simplicity and avoiding over-centralization.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a document or text chunk."""

    id: str = Field(description="Unique document identifier")
    text: str = Field(description="Document text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata",
    )
    score: float | None = Field(
        default=None,
        description="Relevance score if from retrieval",
    )


class QueryRequest(BaseModel):
    """Request model for query processing."""

    query: str = Field(description="User's natural language query")
    context: dict[str, Any] | None = Field(
        default=None,
        description="Optional conversation context",
    )
    use_multi_agent: bool | None = Field(
        default=None,
        description="Override for multi-agent usage",
    )
    retrieval_strategy: Literal["vector", "hybrid", "graphrag"] | None = Field(
        default=None,
        description="Override retrieval strategy",
    )
    top_k: int | None = Field(
        default=None,
        description="Number of documents to retrieve",
        ge=1,
        le=50,
    )


class AgentDecision(BaseModel):
    """Represents an agent's decision in the multi-agent system."""

    agent: str = Field(description="Agent name that made the decision")
    decision_type: str = Field(description="Type of decision made")
    confidence: float = Field(
        description="Confidence score",
        ge=0.0,
        le=1.0,
    )
    reasoning: str | None = Field(
        default=None,
        description="Reasoning behind the decision",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional decision metadata",
    )


class ConversationTurn(BaseModel):
    """Represents a single turn in a conversation."""

    id: str = Field(description="Unique turn identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Turn timestamp",
    )
    role: Literal["user", "assistant", "system"] = Field(
        description="Speaker role",
    )
    content: str = Field(description="Message content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Turn metadata",
    )


class ConversationContext(BaseModel):
    """Manages conversation context for multi-turn interactions."""

    session_id: str = Field(description="Unique session identifier")
    turns: list[ConversationTurn] = Field(
        default_factory=list,
        description="Conversation history",
    )
    total_tokens: int = Field(
        default=0,
        description="Total tokens in context",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Session metadata",
    )

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn to the conversation and update token tracking.

        Args:
            turn (ConversationTurn): The conversation turn to be added.

        Note:
            This is a simplified token counting method. In a production
            environment, a more precise method like tiktoken would be used.
        """
        self.turns.append(turn)
        # Update token count (simplified - would use tiktoken in production)
        self.total_tokens += len(turn.content.split()) * 2

    def get_context_window(self, max_tokens: int = 65536) -> list[ConversationTurn]:
        """Get the most recent conversation turns that fit within a token limit.

        This method retrieves the most recent conversation turns while ensuring
        the total token count does not exceed the specified maximum. The tokens
        are calculated using a simple word-based estimation.

        Args:
            max_tokens (int, optional): Maximum number of tokens allowed in the context.
                Defaults to 65536, which is suitable for many large language models.

        Returns:
            list[ConversationTurn]: A list of conversation turns that fit within
                the token limit, ordered from oldest to newest within the window.

        Note:
            This is a simplified token counting method. In a production environment,
            a more precise tokenization method like tiktoken would be used for
            accurate token counting.
        """
        # Simple implementation - would be more sophisticated in production
        result: list[ConversationTurn] = []
        current_tokens = 0

        for turn in reversed(self.turns):
            turn_tokens = len(turn.content.split()) * 2
            if current_tokens + turn_tokens > max_tokens:
                break
            result.insert(0, turn)
            current_tokens += turn_tokens

        return result


class PerformanceMetrics(BaseModel):
    """Performance metrics for system monitoring."""

    query_latency_ms: float = Field(
        description="Total query processing latency in milliseconds",
    )
    agent_overhead_ms: float = Field(
        description="Agent coordination overhead in milliseconds",
    )
    retrieval_latency_ms: float = Field(
        description="Document retrieval latency in milliseconds",
    )
    llm_latency_ms: float = Field(
        description="LLM generation latency in milliseconds",
    )
    memory_usage_mb: float = Field(
        description="Current memory usage in MB",
    )
    vram_usage_mb: float = Field(
        description="Current VRAM usage in MB",
    )
    tokens_per_second: float = Field(
        description="LLM generation speed in tokens/second",
    )
    cache_hit_rate: float = Field(
        description="Cache hit rate percentage",
        ge=0.0,
        le=100.0,
    )


class ValidationResult(BaseModel):
    """Result from response validation."""

    valid: bool = Field(description="Whether response is valid")
    confidence: float = Field(
        description="Validation confidence",
        ge=0.0,
        le=1.0,
    )
    issues: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Identified issues",
    )
    suggested_action: Literal["accept", "regenerate", "refine"] = Field(
        default="accept",
        description="Suggested action based on validation",
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(description="Error message")
    error_type: str = Field(description="Type of error")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error details",
    )


class PdfPageImageNode(BaseModel):
    """Schema describing a page-image artifact for multimodal reranking.

    Emitted during ingestion for PDF documents with deterministic ID and lineage
    metadata.
    """

    node_id: str = Field(description="Deterministic node identifier (sha256)")
    page_no: int = Field(ge=1, description="1-based page number")
    bbox: tuple[float, float, float, float] = Field(
        description="Page bounding box as (x0,y0,x1,y1)"
    )
    modality: Literal["pdf_page_image"] = Field(
        default="pdf_page_image", description="Modality marker"
    )
    source_path: str = Field(description="Path to source PDF")
    hash: str = Field(description="SHA-256 hash of image bytes")

    traceback: str | None = Field(
        default=None,
        description="Error traceback (debug mode only)",
    )
    suggestion: str | None = Field(
        default=None,
        description="Suggested resolution",
    )


class AnalysisOutput(BaseModel):
    """Structured output schema for document analysis results.

    Defines the expected format for analysis results from the language model,
    ensuring consistent structure for summaries, insights, action items, and
    questions. Used with Pydantic output parsing to validate and structure
    LLM responses.
    """

    summary: str = Field(description="Summary of the document")
    key_insights: list[str] = Field(description="Key insights extracted")
    action_items: list[str] = Field(description="Action items identified")
    open_questions: list[str] = Field(description="Open questions surfaced")
