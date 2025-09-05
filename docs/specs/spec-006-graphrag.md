---
spec: SPEC-006
title: GraphRAG: PropertyGraphIndex with LLMSynonymRetriever
version: 1.0.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-GR-001: Build a PropertyGraphIndex from ingested nodes.
  - FR-GR-002: Provide a graph-aware retriever toggle in UI.
  - NFR-MAINT-002: Persist graph export to GraphML/Parquet.
related_adrs: ["ADR-008","ADR-019"]
---


## Objective

Add optional **GraphRAG** using LlamaIndex **PropertyGraphIndex**. Provide synonyms retriever and a UI toggle to route retrieval through graph.

## Libraries and Imports

```python
from llama_index.indices.property_graph import PropertyGraphIndex
from llama_index.core.retrievers import BaseRetriever
```

## File Operations

### CREATE

- `src/retrieval/graph_config.py`: graph schema, entity/relationship extractors, export functions.
- `src/retrieval/graph_retriever.py`: wrapper to call `index.as_retriever(...)` with synonyms retriever.

### UPDATE

- `src/pages/settings.py`: toggle for GraphRAG; when enabled, chat routes queries through graph-enabled retriever.

## Acceptance Criteria

```gherkin
Feature: GraphRAG toggle
  Scenario: Enable graph-aware retrieval
    Given GraphRAG is enabled in Settings
    When I query
    Then retrieval SHALL call PropertyGraphIndex.as_retriever
```

## References

- LlamaIndex Property Graph docs.
