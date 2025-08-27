"""Models module for DocMind AI.

This module provides access to core data models and analysis schemas.

CONSOLIDATED MODEL ORGANIZATION:
Following the consolidation strategy, models are now organized as:
- Shared models (used across modules) are in schemas.py
- Processing models are consolidated in processing.py
- Embedding models are consolidated in embeddings.py
- Storage models are consolidated in storage.py
- Agent-specific models remain in agents/models.py
"""

# Shared models (used across multiple modules)
# Domain-specific models (consolidated for easy access)
from .embeddings import (
    EmbeddingError,
    EmbeddingParameters,
    EmbeddingResult,
)
from .processing import (
    DocumentElement,
    ProcessingError,
    ProcessingResult,
    ProcessingStrategy,
)
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
from .storage import (
    DocumentMetadata,
    PersistenceError,
    SearchResult,
    StorageStats,
    VectorRecord,
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
    # Processing models
    "DocumentElement",
    "ProcessingResult",
    "ProcessingStrategy",
    "ProcessingError",
    # Embedding models
    "EmbeddingParameters",
    "EmbeddingResult",
    "EmbeddingError",
    # Storage models
    "DocumentMetadata",
    "VectorRecord",
    "SearchResult",
    "StorageStats",
    "PersistenceError",
]
