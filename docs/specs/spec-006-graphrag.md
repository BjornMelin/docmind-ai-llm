---
spec: SPEC-006
title: GraphRAG: PropertyGraphIndex with library-first retriever
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
\n+Update: Align with library-first, documented APIs only. Use `PropertyGraphIndex.as_retriever(...)` directly (no custom synonym retriever), and export graph relations using `property_graph_store.get_rel_map(...)`. Provide opt-in toggle in Settings; default remains off (ADR-019).

## Libraries and Imports

```python
from llama_index.core import PropertyGraphIndex
```

## File Operations

### CREATE / UPDATE

- `src/retrieval/graph_config.py`: graph schema/extractors wiring, traversal helper (thin wrapper on `as_retriever`), and portable exports built from `property_graph_store.get_rel_map(...)` to JSONL/Parquet. May expose a tiny helper wrapper (no index mutation) and a test-only shim for legacy attachment.

### UPDATE

- `src/pages/settings.py`: toggle for GraphRAG; when enabled, chat routes queries through `PropertyGraphIndex.as_retriever(...)`.

## Acceptance Criteria

```gherkin
Feature: GraphRAG toggle
  Scenario: Enable graph-aware retrieval
    Given GraphRAG is enabled in Settings
    When I query
    Then retrieval SHALL call PropertyGraphIndex.as_retriever
  And graph exports SHALL be produced from get_rel_map() to JSONL and (optionally) Parquet
```

## References

- LlamaIndex Property Graph Guide: <https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide>
- Property Graph Examples: <https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_basic>
- Graph store API (`get`, `get_rel_map`, `save_networkx_graph`): see LlamaIndex examples (Context7 snippets)
