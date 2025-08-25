"""Models module for DocMind AI.

This module provides access to core data models and analysis schemas.

Following the hybrid model organization strategy:
- Shared models (used across modules) are centralized in schemas.py
- Domain-specific models remain colocated with their usage domains
- Agent-specific models are in agents.models
"""

# Shared models (used across multiple modules)
from .schemas import (
    AgentDecision,
    AnalysisOutput,
    ConversationContext,
    ConversationTurn,
    Document,
    ErrorResponse,
    PerformanceMetrics,
    QueryRequest,
    ValidationResult,
)

__all__ = [
    # Core data structures
    "Document",
    "QueryRequest",
    # Decision and conversation models
    "AgentDecision",
    "ConversationTurn",
    "ConversationContext",
    # Performance and validation
    "PerformanceMetrics",
    "ValidationResult",
    # Output models
    "AnalysisOutput",
    "ErrorResponse",
]
