# ADR-013: RRF Hybrid Search

## Title

Reciprocal Rank Fusion for Hybrid Search

## Version/Date

4.0 / 2025-01-16

## Status

Accepted

## Description

Implements hybrid search using the native LlamaIndex `HybridFusionRetriever` with Reciprocal Rank Fusion (RRF) as the fusion method. This approach combines results from dense (semantic) and sparse (keyword) retrievers to improve search relevance and is made resilient with `Tenacity`.

## Context

Effective document retrieval requires a combination of two search strategies: dense retrieval, which understands the semantic meaning and intent of a query, and sparse retrieval, which excels at matching specific keywords and acronyms. Relying on only one method leads to suboptimal results. Reciprocal Rank Fusion (RRF) is a proven, score-normalization technique that effectively combines ranked lists from both retrievers into a single, more relevant list. The implementation must be robust against potential transient failures from the underlying vector database.

## Related Requirements

### Functional Requirements

- **FR-1:** The system must combine dense and sparse search results to improve retrieval quality.

### Non-Functional Requirements

- **NFR-1:** **(Maintainability)** The solution must use native LlamaIndex components to minimize custom code.
- **NFR-2:** **(Resilience)** The retrieval process must be resilient to transient errors from the vector store.

### Performance Requirements

- **PR-1:** The fusion process should not add significant latency to the query pipeline.

### Integration Requirements

- **IR-1:** The retriever must be configurable via the global `Settings` singleton.
- **IR-2:** The retriever must support asynchronous execution within the `QueryPipeline`.

## Alternatives

### 1. Simple Score Averaging

- **Description**: Combine results by adding or averaging the raw scores from the dense and sparse retrievers.
- **Issues**: Dense and sparse scores are not on a comparable scale, leading to poor and unpredictable ranking when naively combined.
- **Status**: Rejected.

### 2. Custom Fusion Logic

- **Description**: Write custom Python code to merge and re-rank the two result sets.
- **Issues**: Violates the library-first principle and is likely to be less performant and robust than the native, battle-tested implementation in LlamaIndex.
- **Status**: Rejected.

## Decision

We will adopt the native LlamaIndex **`HybridFusionRetriever`** as the standard component for hybrid search. It will be configured to use **`"rrf"` (Reciprocal Rank Fusion)** mode. The entire retrieval operation will be wrapped in a **`Tenacity`** retry decorator to ensure resilience against transient vector database connection errors, as established in `ADR-022`.

## Related Decisions

- **ADR-002** (Embedding Choices): Provides the dense (BGE) and sparse (SPLADE++) embeddings that power the underlying retrievers.
- **ADR-006** (Analysis Pipeline): The `HybridFusionRetriever` is the first component in the main `QueryPipeline`.
- **ADR-020** (LlamaIndex Native Settings Migration): The retriever's parameters are configured via the `Settings` singleton.
- **ADR-012** (Async Performance Optimization): The retriever is executed asynchronously via `QueryPipeline.arun()`.
- **ADR-022** (Tenacity Resilience Integration): Provides the resilience pattern for this component.

## Design

### Architecture Overview

The `HybridFusionRetriever` acts as a meta-retriever, dispatching the query to two underlying retrievers (one for dense vectors, one for sparse) and then fusing the results using RRF.

```mermaid
graph TD
    A[Query] --> B[HybridFusionRetriever];
    B --> C[Dense Retriever (Vector Store)];
    B --> D[Sparse Retriever (Vector Store)];
    C --> E[Dense Results];
    D --> F[Sparse Results];
    E --> G[RRF Fusion Engine];
    F --> G;
    G --> H[Final Ranked Results];
```

### Implementation Details

**In `retriever_factory.py`:**

```python
# This code demonstrates the resilient, async retriever setup
from llama_index.core.retrievers import VectorIndexRetriever, HybridFusionRetriever
from llama_index.core import Settings, VectorStoreIndex
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

# Assume dense_index and sparse_index are already created VectorStoreIndex objects
dense_retriever = VectorIndexRetriever(index=dense_index, similarity_top_k=10)
sparse_retriever = VectorIndexRetriever(index=sparse_index, similarity_top_k=10)

# The meta-retriever that combines the two
hybrid_retriever = HybridFusionRetriever(
    dense_retriever,
    sparse_retriever,
    mode="rrf",
    # The alpha parameter controls the weighting, configurable via Settings
    similarity_top_k=Settings.retriever_top_k or 10
)

# ADR-022: Apply Tenacity for resilience against vector DB errors
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    # Add specific DB connection error types here
    # e.g., retry=retry_if_exception_type(DBConnectionError)
)
async def resilient_retrieve(query_text: str):
    """
    Executes a hybrid retrieval query asynchronously with a resilience wrapper.
    """
    logger.info(f"Performing resilient hybrid retrieval for query: '{query_text}'")
    try:
        # aretrieve is the async method for the retriever
        return await hybrid_retriever.aretrieve(query_text)
    except Exception as e:
        logger.error(f"Hybrid retrieval failed after multiple retries: {e}")
        raise
```

## Testing

**In `tests/test_retriever.py`:**

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_resilient_retrieve_handles_errors():
    """
    Verify that the Tenacity wrapper correctly retries on failure
    and eventually raises an exception.
    """
    # Mock the retriever's async method to always fail
    mock_retriever = AsyncMock()
    mock_retriever.aretrieve.side_effect = ConnectionError("DB not available")
    
    # Patch the hybrid_retriever in the factory module
    with patch('retriever_factory.hybrid_retriever', mock_retriever):
        from retriever_factory import resilient_retrieve
        
        with pytest.raises(ConnectionError):
            await resilient_retrieve("test query")
            
        # Assert that the mock was called 3 times (initial call + 2 retries)
        assert mock_retriever.aretrieve.call_count == 3
```

## Consequences

### Positive Outcomes

- **Improved Relevance**: Combining dense and sparse search provides demonstrably better retrieval results than either method alone, improving the overall quality of the RAG system.
- **Increased Resilience**: The `Tenacity` wrapper makes the core retrieval function robust against transient infrastructure issues, improving system uptime and reliability.
- **High Maintainability**: Using the native `HybridFusionRetriever` eliminates custom fusion logic and aligns with the library-first principle.

### Negative Consequences / Trade-offs

- **Increased Latency**: A hybrid query is marginally slower than a single-vector query because it performs two searches. This trade-off is acceptable for the significant gain in relevance.

### Ongoing Maintenance & Considerations

- **Vector Database Monitoring**: The health of the underlying vector database (Qdrant) is critical. It should be monitored for performance and availability.

### Dependencies

- **Python**: `llama-index-core>=0.12.0`, `tenacity>=8.2.0`

## Changelog

- **4.0 (2025-01-16)**: Integrated the resilience pattern from ADR-022 by adding a `Tenacity` retry decorator to the core retrieval function. Aligned all code with the `Settings` singleton and async patterns.
- **3.0 (2025-01-13)**: Integrated `Settings.embed_model` GPU optimization. Added `QueryPipeline.parallel_run()` async patterns.
- **2.0 (2025-07-25)**: Switched to `HybridFusionRetriever`; Added alpha/prefetch toggle/integration with pipeline.
