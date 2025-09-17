"""Models module for DocMind AI.

Expose Pydantic models used across the application.
"""

from .embeddings import EmbeddingParameters, EmbeddingResult
from .processing import (
    ExportArtifact,
    IngestionConfig,
    IngestionInput,
    IngestionResult,
    ManifestSummary,
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
    "AgentDecision",
    "AnalysisOutput",
    "ConversationContext",
    "ConversationTurn",
    "Document",
    "DocumentMetadata",
    "EmbeddingParameters",
    "EmbeddingResult",
    "ErrorResponse",
    "ExportArtifact",
    "IngestionConfig",
    "IngestionInput",
    "IngestionResult",
    "ManifestSummary",
    "PerformanceMetrics",
    "PersistenceError",
    "QueryRequest",
    "SearchResult",
    "StorageStats",
    "ValidationResult",
    "VectorRecord",
]
