"""Custom exception classes with contextual error handling for DocMind AI.

This module provides domain-specific exception classes with rich context information
for comprehensive error tracking and debugging. Each exception automatically logs
its context and provides structured error information for better error handling.

Features:
- Domain-specific exception hierarchy
- Automatic context logging with loguru
- Structured error information
- Original exception chaining
- Performance impact tracking
- Security-conscious error messages

Example:
    Basic usage::

        from utils.exceptions import EmbeddingError, IndexCreationError

        try:
            create_embeddings(docs)
        except Exception as e:
            raise EmbeddingError(
                "Failed to generate embeddings",
                context={"doc_count": len(docs), "model": "bge-large"},
                original_error=e
            )

Classes:
    DocMindError: Base exception for all DocMind AI errors
    EmbeddingError: Embedding generation and processing errors
    IndexCreationError: Vector/KG index creation errors
    DocumentLoadingError: Document parsing and loading errors
    ConfigurationError: Settings and configuration errors
    ResourceError: Resource management and cleanup errors
    AgentError: Agent creation and execution errors
"""

import time
from typing import Any

from utils.logging_config import logger


class DocMindError(Exception):
    """Base exception for all DocMind AI errors.

    Provides structured error handling with automatic logging, context preservation,
    and performance impact tracking. All DocMind exceptions should inherit from
    this base class for consistent error handling.

    Attributes:
        context: Dictionary of contextual information about the error
        original_error: The original exception that caused this error (if any)
        timestamp: When the error occurred
        operation: Name of the operation that failed

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     raise DocMindError(
        ...         "Operation failed",
        ...         context={"operation": "indexing", "retry_count": 3},
        ...         original_error=e
        ...     )
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
        operation: str | None = None,
        **kwargs,
    ):
        """Initialize DocMind error with context and logging.

        Args:
            message: Human-readable error message
            context: Dictionary of contextual information
            original_error: Original exception that caused this error
            operation: Name of the operation that failed
            **kwargs: Additional context fields
        """
        super().__init__(message)

        self.context = context or {}
        self.context.update(kwargs)  # Add any extra context
        self.original_error = original_error
        self.timestamp = time.time()
        self.operation = operation or self.context.get("operation", "unknown")

        # Log the error with full context
        self._log_error(message)

    def _log_error(self, message: str) -> None:
        """Log the error with structured context."""
        error_data = {
            "error_class": self.__class__.__name__,
            "operation": self.operation,
            "context": self.context,
            "timestamp": self.timestamp,
        }

        if self.original_error:
            error_data["original_error"] = {
                "type": type(self.original_error).__name__,
                "message": str(self.original_error),
            }

        logger.error(
            f"{self.__class__.__name__}: {message}",
            extra={"error_details": error_data},
            exception=self.original_error if self.original_error else self,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert error to structured dictionary for API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "operation": self.operation,
            "context": self.context,
            "timestamp": self.timestamp,
            "original_error": (
                {
                    "type": type(self.original_error).__name__,
                    "message": str(self.original_error),
                }
                if self.original_error
                else None
            ),
        }

    def __str__(self) -> str:
        """String representation with operation context."""
        base_msg = super().__str__()
        if self.operation != "unknown":
            return f"[{self.operation}] {base_msg}"
        return base_msg


class EmbeddingError(DocMindError):
    """Error during embedding generation or processing.

    Raised when embedding models fail to generate vectors, encounter
    hardware issues, or experience configuration problems. Includes
    specific context about model parameters and input data.

    Example:
        >>> raise EmbeddingError(
        ...     "CUDA out of memory during embedding",
        ...     context={
        ...         "model": "bge-large-en-v1.5",
        ...         "batch_size": 32,
        ...         "gpu_memory_gb": 8,
        ...         "input_tokens": 50000
        ...     },
        ...     operation="batch_embedding"
        ... )
    """

    pass


class IndexCreationError(DocMindError):
    """Error during vector or knowledge graph index creation.

    Raised when index creation fails due to Qdrant connectivity issues,
    embedding dimension mismatches, or knowledge graph extraction problems.

    Example:
        >>> raise IndexCreationError(
        ...     "Failed to create hybrid index",
        ...     context={
        ...         "index_type": "hybrid_vector",
        ...         "doc_count": 1000,
        ...         "embedding_dim": 1024,
        ...         "qdrant_url": "http://localhost:6333"
        ...     },
        ...     operation="index_creation"
        ... )
    """

    pass


class DocumentLoadingError(DocMindError):
    """Error during document parsing and loading.

    Raised when document loaders fail to parse files, extract content,
    or handle multimodal elements. Includes file format and parsing details.

    Example:
        >>> raise DocumentLoadingError(
        ...     "PDF parsing failed - corrupted file",
        ...     context={
        ...         "file_path": "/docs/report.pdf",
        ...         "file_size_mb": 15.2,
        ...         "parser": "unstructured",
        ...         "multimodal": True
        ...     },
        ...     operation="document_parsing"
        ... )
    """

    pass


class ConfigurationError(DocMindError):
    """Configuration validation or loading error.

    Raised when application settings are invalid, missing required values,
    or contain incompatible configurations. Critical for startup validation.

    Example:
        >>> raise ConfigurationError(
        ...     "Invalid embedding model configuration",
        ...     context={
        ...         "model_name": "invalid/model",
        ...         "expected_dim": 1024,
        ...         "actual_dim": 512,
        ...         "config_source": ".env"
        ...     },
        ...     operation="config_validation"
        ... )
    """

    pass


class ResourceError(DocMindError):
    """Resource management or cleanup error.

    Raised when resource allocation, deallocation, or management fails.
    Includes GPU memory issues, connection pool problems, or file I/O errors.

    Example:
        >>> raise ResourceError(
        ...     "Failed to acquire Qdrant connection from pool",
        ...     context={
        ...         "pool_size": 10,
        ...         "active_connections": 8,
        ...         "timeout_seconds": 30,
        ...         "qdrant_url": "http://localhost:6333"
        ...     },
        ...     operation="connection_pooling"
        ... )
    """

    pass


class AgentError(DocMindError):
    """Agent creation or execution error.

    Raised when ReAct agents fail to initialize, execute queries, or
    manage conversation memory. Includes LLM backend and tool information.

    Example:
        >>> raise AgentError(
        ...     "Agent query execution failed",
        ...     context={
        ...         "agent_type": "ReActAgent",
        ...         "llm_backend": "ollama",
        ...         "model": "llama2:7b",
        ...         "tools": ["hybrid_search", "kg_query"],
        ...         "query_length": 150
        ...     },
        ...     operation="agent_query"
        ... )
    """

    pass


class RetryExhaustedError(DocMindError):
    """All retry attempts have been exhausted.

    Raised by the retry mechanism when all configured retry attempts
    have failed. Contains information about retry strategy and attempts.

    Example:
        >>> raise RetryExhaustedError(
        ...     "Operation failed after all retries",
        ...     context={
        ...         "max_attempts": 5,
        ...         "total_duration": 45.6,
        ...         "retry_strategy": "exponential_backoff",
        ...         "last_error": "ConnectionTimeout"
        ...     },
        ...     operation="index_creation_with_retries"
        ... )
    """

    pass


class ValidationError(DocMindError):
    """Input validation or data integrity error.

    Raised when input data fails validation checks, has incorrect formats,
    or violates business rules. Used for request validation and data quality.

    Example:
        >>> raise ValidationError(
        ...     "Invalid document format for processing",
        ...     context={
        ...         "file_type": "unknown",
        ...         "file_size": 0,
        ...         "expected_types": ["pdf", "docx", "txt"],
        ...         "validation_rule": "non_empty_supported_format"
        ...     },
        ...     operation="document_validation"
        ... )
    """

    pass


class PerformanceError(DocMindError):
    """Performance threshold violation or resource exhaustion.

    Raised when operations exceed performance thresholds, consume too many
    resources, or fail to meet SLA requirements. Includes performance metrics.

    Example:
        >>> raise PerformanceError(
        ...     "Embedding generation exceeded time limit",
        ...     context={
        ...         "duration_seconds": 300,
        ...         "threshold_seconds": 120,
        ...         "batch_size": 100,
        ...         "gpu_utilization": 95,
        ...         "memory_usage_gb": 12.5
        ...     },
        ...     operation="batch_embedding"
        ... )
    """

    pass


# Exception hierarchy for easy catching
class CriticalError(DocMindError):
    """Critical system error requiring immediate attention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Critical errors get elevated logging
        logger.critical(
            f"CRITICAL ERROR: {str(self)}",
            extra={
                "error_details": self.to_dict(),
                "requires_immediate_attention": True,
            },
        )


class RecoverableError(DocMindError):
    """Error that might be resolved with retries or fallback strategies."""

    pass


# Convenience functions for common error patterns
def handle_agent_error(
    error: Exception, operation: str = "agent_execution", **context
) -> AgentError:
    """Convert generic exception to AgentError with context."""
    return AgentError(
        f"Agent operation failed: {str(error)}",
        context=context,
        original_error=error,
        operation=operation,
    )


def handle_embedding_error(
    error: Exception, operation: str = "embedding_generation", **context
) -> EmbeddingError:
    """Convert generic exception to EmbeddingError with context."""
    return EmbeddingError(
        f"Embedding operation failed: {str(error)}",
        context=context,
        original_error=error,
        operation=operation,
    )


def handle_index_error(
    error: Exception, operation: str = "index_creation", **context
) -> IndexCreationError:
    """Convert generic exception to IndexCreationError with context."""
    return IndexCreationError(
        f"Index operation failed: {str(error)}",
        context=context,
        original_error=error,
        operation=operation,
    )


def handle_document_error(
    error: Exception, operation: str = "document_loading", **context
) -> DocumentLoadingError:
    """Convert generic exception to DocumentLoadingError with context."""
    return DocumentLoadingError(
        f"Document operation failed: {str(error)}",
        context=context,
        original_error=error,
        operation=operation,
    )


# Export all exception classes and utilities
__all__ = [
    "DocMindError",
    "EmbeddingError",
    "IndexCreationError",
    "DocumentLoadingError",
    "ConfigurationError",
    "ResourceError",
    "AgentError",
    "RetryExhaustedError",
    "ValidationError",
    "PerformanceError",
    "CriticalError",
    "RecoverableError",
    "handle_agent_error",
    "handle_embedding_error",
    "handle_index_error",
    "handle_document_error",
]
