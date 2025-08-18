# ADR-014: BGE-reranker-v2-m3 Reranking Strategy

## Title

Standardized Reranking with BGE-reranker-v2-m3

## Version/Date

8.0 / 2025-01-16

## Status

Accepted

## Description

Adopts `BAAI/bge-reranker-v2-m3` as the single, standardized reranking model for the v1.0 release. This model is implemented via the native `SentenceTransformerRerank` postprocessor and was chosen for its optimal balance of high performance, low resource usage, and native multimodal reranking capability. This decision supersedes all previous reranking strategies.

## Context

Post-retrieval reranking is a critical step for improving context quality. The system requires a single, default reranker that is fast, memory-efficient, and can handle the project's multimodal contexts (text from documents, tables, and image captions). `BGE-reranker-v2-m3` is the ideal choice, providing an 86% reduction in model size and a 3-5x inference speedup compared to larger alternatives, while fully supporting the system's functional requirements.

## Related Requirements

- **Unified Multimodal Reranking**: The model must effectively rerank nodes containing text from any source.
- **Lightweight & Performant**: The model must have a small memory footprint (<1GB) and not be a query bottleneck.
- **LlamaIndex Native**: The implementation must use a native LlamaIndex postprocessor.
- **GPU Optimization**: The reranker must leverage available GPUs automatically via `device_map="auto"`.
- **Async Processing**: The reranker must support asynchronous execution within the `QueryPipeline`.

## Alternatives

### Evaluated Models (Decision Framework Scoring)

| Model                  | Solution Leverage (35%) | Application Value (30%) | Maintenance (25%) | Adaptability (10%) | Total Score | Decision      |
| ---------------------- | ----------------------- | ----------------------- | ----------------- | ------------------ | ----------- | ------------- |
| **BGE-reranker-v2-m3** | 0.90                    | 0.75                    | 0.95              | 0.80               | **0.8625**  | ✅ **Selected** |
| mxbai-rerank-base-v2   | 0.85                    | 0.80                    | 0.80              | 0.85               | 0.8225      | Secondary     |
| ColBERT v2 (text-only) | 0.70                    | 0.85                    | 0.60              | 0.70               | 0.7425      | Text fallback |
| Jina m0 (Superseded)   | 0.70                    | 0.95                    | 0.30              | 0.40               | 0.6175      | Rejected      |

### Rejected Alternative: Dual-Reranker Strategy

A previous iteration of this ADR considered offering a user-selectable fallback reranker (ColBERT). This was **rejected** for the v1.0 release as it introduced unnecessary complexity in the UI, state management, and resource loading, violating the project's core KISS (Keep It Simple, Stupid) principle.

## Decision

The architecture will standardize on a single reranking model for v1.0: **`BAAI/bge-reranker-v2-m3`**. It will be instantiated once, configured globally on the `llama_index.core.Settings` singleton, and used for all reranking tasks. This approach maximizes simplicity, stability, and performance for the initial release.

## Related Decisions

- `ADR-007` (Reranking Strategy): This ADR is superseded by this decision.
- `ADR-006` (Analysis Pipeline): The reranker is a key component in the `QueryPipeline`.
- `ADR-003` (GPU Optimization): Leverages `device_map="auto"` for seamless GPU acceleration.
- `ADR-012` (Async Performance Optimization): Reranking is executed asynchronously via `QueryPipeline.arun()`.
- `ADR-023` (PyTorch Optimization Strategy): Leverages mixed precision (`torch_dtype="float16"`).
- `ADR-020` (LlamaIndex Native Settings Migration): All configuration is handled by the native `Settings` singleton.

## Design

### Global Reranker Configuration

The design is exceptionally simple. The reranker is instantiated and set on the global `Settings` object during application startup.

```python
# In application_setup.py
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import Settings
import torch
import logging

logger = logging.getLogger(__name__)

def configure_global_components():
    """Configures and sets all global LlamaIndex components."""
    
    # ... other settings ...

    logger.info("Initializing reranker: BAAI/bge-reranker-v2-m3")
    Settings.reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=Settings.reranking_top_k or 5,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # ... other settings ...
```

### Query Pipeline Integration

The `QueryPipeline` is simplified as it no longer needs to account for multiple rerankers. It directly and reliably uses the globally configured instance from `Settings`.

```python
# In pipeline_factory.py
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import Settings

def create_analysis_pipeline() -> QueryPipeline:
    """Creates the main query pipeline, using the globally set reranker."""
    return QueryPipeline(
        chain=[
            hybrid_retriever,
            Settings.reranker, # Directly uses the single, globally configured reranker
            response_synthesizer
        ],
        async_mode=True,
        verbose=True
    )
```

## Consequences

### Positive Outcomes

- **Maximum Simplicity**: The removal of the dual-reranker strategy eliminates significant code complexity related to factories, UI state, and dynamic pipeline generation.
- **Reduced Resource Usage**: The system only ever needs to load one reranker model, preserving VRAM for other components.
- **Improved Stability and Maintainability**: With fewer moving parts and a single, clear path for reranking, the system is more robust and easier to debug.
- **Excellent Performance**: The chosen BGE model provides a 3-5x speedup over larger models, ensuring a responsive user experience.
- **Full Capability for v1.0**: The selected model handles all required reranking tasks, including multimodal contexts, meeting all core requirements.

### Future Considerations

- If strong user feedback after the v1.0 release indicates a demand for a higher-accuracy, text-only reranking mode, the dual-reranker strategy can be re-evaluated and introduced as a new feature in a future version.

## Changelog

- **8.0 (2025-01-16)**: Finalized decision to simplify to a single, default reranker (`BGE-reranker-v2-m3`) for the v1.0 release. Removed all logic for a fallback model and user-configurable toggle to maximize simplicity, stability, and resource efficiency.
- **7.0 (2025-01-15)**: Added a user-configurable selection mechanism for the fallback reranker. (This approach was superseded by version 8.0).
- **6.0 (2025-01-15)**: Restored and enhanced async integration details and testing strategy.
- **5.1 (2025-01-14)**: Updated to better emphasize the multimodal capabilities of the BGE reranker.
- **5.0 (2025-01-14)**: Rewritten to be the authoritative ADR for reranking.
- **4.0 (2025-01-13)**: Major revision. Replaced Jina m0 with BGE-reranker-v2-m3 as primary.
- **3.0 (2025-01-13)**: Integrated GPU optimization and async patterns.
- **2.0 (2025-07-25)**: Integrated with QueryPipeline.
