---
ADR: 003
Title: Adaptive Retrieval Pipeline with RAPTOR‑Lite
Status: Accepted
Version: 3.5
Date: 2025-09-04
Supersedes:
Superseded-by:
Related: 001, 002, 004, 009, 010, 011, 019, 024, 030, 037
Tags: architecture, retrieval, routing, raptor-lite, hybrid, reranking, local-first
References:
- [RouterQueryEngine — LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine/)
- [Retriever Modules (Hybrid/Multi-Query) — LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/retrievers/)
- [Qdrant Metadata Filter Example — LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/)
- [TokenCountingHandler — LlamaIndex](https://docs.llamaindex.ai/en/v0.10.22/examples/callbacks/TokenCountingHandler/)
---

## Description

Adopts a library‑first adaptive retrieval pipeline that combines RAPTOR‑Lite hierarchical organization with query‑aware routing. The pipeline selects vector, hybrid, or multi‑query retrieval at runtime and optionally augments with GraphRAG and DSPy query optimization, maximizing quality while staying simple and local‑first.

## Context

Flat vector search limits multi‑level synthesis and robustness. We need:

- Hierarchical context for complex questions (detail + overview)
- Strategy adaptation per query (vector/hybrid/decomposition)
- Corrective behaviors when initial retrieval underperforms
- Local‑first efficiency compatible with consumer hardware and 128K context

Full RAPTOR is too heavy for local use. RAPTOR‑Lite preserves hierarchical value while leveraging LlamaIndex built‑ins (router, hybrid, multi‑query, filters) to keep complexity and maintenance low. The pipeline consumes BGE‑M3 (ADR‑002) in Qdrant (ADR‑031/030), aligns with 128K context (ADR‑004/010), and applies modality‑aware reranking (ADR‑037).

## Decision Drivers

- Library‑first primitives over custom implementations
- Simplicity and maintainability for local‑first deployment
- Multimodal support with clear fallbacks
- Deterministic performance and predictable resource footprint

## Alternatives

- A: Flat vector + rerank — Pros: simple, fast; Cons: weak synthesis/robustness
- B: Full RAPTOR — Pros: strong hierarchy; Cons: resource‑heavy, complex
- C: RAPTOR‑Lite + router (chosen) — Pros: balanced quality/perf, low complexity; Cons: small routing overhead

### Decision Framework

| Model / Option                | Solution Leverage (35%) | Application Value (30%) | Maintenance (25%) | Adaptability (10%) | Total Score | Decision       |
| ---------------------------- | ----------------------- | ----------------------- | ----------------- | ------------------ | ----------- | -------------- |
| **RAPTOR‑Lite + Router**     | 8.5                     | 8.0                     | 8.0               | 9.0                | **8.3**     | ✅ Selected     |
| Flat Vector + Rerank         | 6.0                     | 6.5                     | 9.0               | 5.0                | 6.8         | Rejected       |
| Full RAPTOR                  | 9.0                     | 8.0                     | 3.0               | 8.5                | 6.9         | Rejected       |

## Decision

We will adopt RouterQueryEngine with HybridRetriever and MultiQueryRetriever as the default adaptive retrieval stack. Configuration: `similarity_top_k=10`, hybrid fusion via RRF, multi‑query `num_queries=3`. DSPy query optimization (ADR‑018) and PropertyGraphIndex (ADR‑019) are optional augmentations. This consolidates prior ad‑hoc routing into a single library‑first component.

## High-Level Architecture

```mermaid
graph TD
    A[Documents] --> B[Chunking (ADR-009)]
    B --> C[RAPTOR‑Lite Sections]
    C --> D[Section Summaries]
    D --> E[Qdrant Index]

    Q[User Query] --> R{RouterQueryEngine}
    R -->|Vector| V[Vector Retriever]
    R -->|Hybrid| H[Hybrid Retriever (RRF)]
    R -->|Decompose| M[Multi‑Query Retriever]
    R -->|Graph (opt)| G[PropertyGraphIndex]

    V --> X[Results]
    H --> X
    M --> X
    G --> X
    X --> Y[Reranker (ADR‑037)] --> Z[Answer]
```

## Related Requirements

### Functional Requirements

- FR-1: Route queries to the optimal retrieval strategy automatically.
- FR-2: Provide hierarchical access to both detail‑level chunks and section summaries.
- FR-3: Retry with alternate strategies when relevance/coverage is low.
- FR-4: Support both focused fact lookup and broad synthesis.

### Non-Functional Requirements

- NFR-1 (Maintainability): Prefer built‑ins; <200 LOC of glue code.
- NFR-2 (Local‑First): All core retrieval runs offline on consumer hardware.
- NFR-3 (Quality): ≥15% improvement on complex queries vs flat vector baseline.

### Performance Requirements

- PR-1: P95 end‑to‑end query latency ≤ 2s on target hardware including reranking.
- PR-2: Routing overhead ≤ 300ms; additional hierarchy overhead ≤ 500ms.

### Integration Requirements

- IR-1: Integrate with LlamaIndex `Settings` for models and callbacks.
- IR-2: Provide async interface compatible with `async/await`.

## Design

### Architecture Overview

See High‑Level Architecture diagram. RAPTOR‑Lite hierarchy is produced upstream (ADR‑009); retrieval focuses on routing and strategy selection using LlamaIndex primitives.

### Implementation Details

```python
# Minimal, library-first setup
from llama_index.core import Settings
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.retrievers import HybridRetriever, MultiQueryRetriever
from llama_index.core.tools import QueryEngineTool

def build_adaptive_query_engine(index, llm, enable_dspy: bool = False, enable_graph: bool = False):
    base_retriever = index.as_retriever(similarity_top_k=10)

    hybrid = HybridRetriever(
        vector_retriever=index.as_retriever(similarity_top_k=5),
        keyword_retriever=index.as_retriever(mode="keyword", top_k=5),
        fusion_mode="reciprocal_rank",
    )

    multi = MultiQueryRetriever.from_defaults(retriever=hybrid, llm=llm, num_queries=3)

    tools = [
        QueryEngineTool.from_defaults(base_retriever.as_query_engine(), name="vector_search",
                                      description="Semantic similarity for factual queries"),
        QueryEngineTool.from_defaults(hybrid.as_query_engine(), name="hybrid_search",
                                      description="Keyword + vector for recall/precision"),
        QueryEngineTool.from_defaults(multi.as_query_engine(), name="multi_query",
                                      description="Decompose complex questions"),
    ]

    if enable_graph:
        from src.graphrag_integration import OptionalGraphRAG
        tools.append(
            QueryEngineTool.from_defaults(
                OptionalGraphRAG(enabled=True, vector_store=index).as_query_engine(),
                name="graph_search",
                description="Relationship/multi-hop queries",
            )
        )

    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=tools,
        verbose=True,
    )
```

### Configuration

```env
DOCMIND_RETRIEVAL__ROUTER_VERBOSE=true
DOCMIND_RETRIEVAL__MULTI_QUERY=3
DOCMIND_RETRIEVAL__TOP_K=10
```

## Testing

```python
import time
import pytest

@pytest.mark.asyncio
async def test_latency_budget(async_engine):
    start = time.monotonic()
    out = await async_engine.aquery("Where is the architecture described?")
    assert out is not None
    assert time.monotonic() - start <= 2.0  # PR-1

def test_config_toggles(settings):
    assert settings.retrieval.multi_query == 3
```

## Consequences

### Positive Outcomes

- Improves complex‑query answer quality while remaining local‑first.
- Reduces custom code by relying on LlamaIndex primitives.
- Enables modality‑aware reranking (ADR‑037) and optional GraphRAG.

### Negative Consequences / Trade-offs

- Adds small routing latency overhead (≤300ms).
- RAPTOR‑Lite summaries increase storage modestly.

### Ongoing Maintenance & Considerations

- Review LlamaIndex releases for retriever/router updates.
- Re‑tune `top_k`, RRF, and `num_queries` quarterly.
- Track latency and quality metrics; revisit thresholds if P95 drifts.

### Dependencies

- System: Qdrant (local or remote)
- Python: `llama-index>=0.10`, `qdrant-client>=1.6`, `FlagEmbedding>=1.2`
- Removed: custom router/hybrid code paths (replaced with built‑ins)

## Changelog

- 3.5 (2025-09-04): Rewrote to match ADR template; clarified requirements, decision framework, and testing; pruned legacy code blocks.
- 3.4 (2025‑09‑04): Integrated multimodal reranking (ADR‑037); aligned with 128K context (ADR‑004/010).
- 3.2 (2025-09-02): Standardized to 128K context; updated cache references (ADR‑030).
- 3.1 (2025-08-21): Implementation complete with RouterQueryEngine.
- 2.1 (2025-08-18): Added DSPy optimization and PropertyGraphIndex option.
- 2.0 (2025-08-17): Multi‑strategy routing baseline.
- 1.0 (2025-01-16): Initial RAPTOR‑Lite concept.
