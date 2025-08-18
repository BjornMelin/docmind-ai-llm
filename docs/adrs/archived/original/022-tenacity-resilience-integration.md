# ADR-022: Tenacity Resilience Integration

## Title

Tenacity Integration for Production-Grade Resilience

## Version/Date

2.0 / 2025-01-16

## Status

Accepted

## Description

Implements the `Tenacity` library to provide production-grade resilience for critical infrastructure operations. This strategy complements the native LlamaIndex retry mechanisms by adding robust error handling, with exponential backoff, for operations like database connections, file I/O, and initial model downloads.

## Context

An offline-first application is not immune to transient failures. File systems can have temporary read/write errors, a local vector database might be slow to start, or the initial, one-time download of models from HuggingFace can be interrupted by a flaky network connection. The native retry mechanisms within LlamaIndex are primarily focused on LLM API calls and are not designed to cover these infrastructure-level failure points. A dedicated, robust retry mechanism is required to make the application production-ready.

## Related Requirements

### Non-Functional Requirements

- **NFR-1:** **(Resilience)** The system must automatically recover from transient infrastructure failures (e.g., file I/O errors, temporary database unavailability).
- **NFR-2:** **(Maintainability)** The resilience logic must be implemented using a standard, well-maintained library, not custom code.

### Integration Requirements

- **IR-1:** The solution must complement, not conflict with, the existing retry logic within LlamaIndex.

## Alternatives

### 1. LlamaIndex Native Retries Only

- **Description**: Rely solely on the built-in retry mechanisms within LlamaIndex.
- **Issues**: This provides insufficient coverage. LlamaIndex's retries are scoped to specific components (like LLM calls) and do not cover infrastructure operations like database connections or file reads.
- **Status**: Rejected.

### 2. Custom Retry Implementation

- **Description**: Write custom `try...except` loops with `time.sleep()` to handle retries.
- **Issues**: This is a classic violation of the library-first principle. It leads to boilerplate code and lacks sophisticated strategies like exponential backoff or jitter.
- **Status**: Rejected.

## Decision

We will adopt the **`Tenacity`** library as the standard for implementing resilience for all critical infrastructure operations. We will use its decorator-based approach to wrap functions that interact with the file system, the vector database, and the initial model download process. This creates a "Hybrid Resilience" model where LlamaIndex handles application-level retries and `Tenacity` handles infrastructure-level retries.

## Related Decisions

- **ADR-021** (LlamaIndex Native Architecture Consolidation): This decision to use a strategic external library fills an identified gap in the native LlamaIndex ecosystem.
- **ADR-018** (Refactoring Decisions): Continues the library-first approach by adopting a best-in-class library for a specific problem.
- **ADR-004** (Document Loading): The `Tenacity` resilience pattern is applied to the document parsing and loading workflow.
- **ADR-013** (RRF Hybrid Search): The `Tenacity` resilience pattern is applied to the core vector store retrieval operations.
- **ADR-002** (Embedding Choices): The `Tenacity` resilience pattern is applied to the initial, one-time download of the embedding models from online hubs.

## Design

### Architecture Overview: Hybrid Resilience

The system employs a two-layer approach to resilience.

```mermaid
graph TD
    A[User Action] --> B[Application Logic];
    B --> C{Interaction Type?};
    C -->|Infrastructure<br/>(DB Query, File Read)| D["Tenacity Retry Decorator<br/>(Exponential Backoff)"];
    C -->|LlamaIndex Core<br/>(LLM Call)| E["Native LlamaIndex Retry<br/>(Fixed Retries)"];
    D --> F[External System];
    E --> F;
```

### Implementation Details

**Example 1: Resilient Document Loading (from `ADR-004`)**

```python
# In document_processor.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((IOError, OSError))
)
async def load_and_process_document(file_path: str):
    # ... document loading logic ...
    pass
```

**Example 2: Resilient Vector Search (from `ADR-013`)**

```python
# In retriever_factory.py
from tenacity import retry, stop_after_attempt, wait_exponential

# Assume DBConnectionError is a specific exception from the DB client
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
    # retry=retry_if_exception_type(DBConnectionError)
)
async def resilient_retrieve(query_text: str):
    # ... hybrid_retriever.aretrieve(query_text) logic ...
    pass
```

**Example 3: Resilient Model Download (from `ADR-002`)**

```python
# In application_setup.py
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30)
)
def configure_embedding_models():
    # ... Settings.embed_model = HuggingFaceEmbedding(...) logic ...
    pass
```

## Consequences

### Positive Outcomes

- **Improved Reliability**: The application can now automatically recover from the most common transient infrastructure failures, leading to a more stable user experience.
- **Production Readiness**: Implementing a robust resilience strategy is a key step in making the application suitable for production use.
- **Clean Code**: Using decorators keeps the resilience logic separate from the core business logic, making the code easier to read and maintain.

### Negative Consequences / Trade-offs

- **Added Dependency**: Introduces a new dependency on the `tenacity` library. This is a well-justified trade-off for the significant gain in reliability.
- **Masked Failures**: If not configured carefully, a retry mechanism can mask a persistent underlying problem. The logging within the retry functions is critical for monitoring.

### Dependencies

- **Python**: `tenacity>=8.2.0`

## Changelog

- **2.0 (2025-01-16)**: Finalized as the definitive resilience strategy. Aligned all code examples with the final architecture and explicitly defined the "Hybrid Resilience" model.
- **1.0 (2025-01-13)**: Initial implementation of Tenacity resilience integration.
