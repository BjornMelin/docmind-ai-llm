---
spec: SPEC-006
title: GraphRAG: PropertyGraphIndex with Router and Library‑First Helpers
version: 1.1.0
date: 2025-09-09
owners: ["ai-arch"]
status: Revised
related_requirements:
  - FR-009.1: Router engine wiring (vector+graph; fallback)
  - FR-009.3: Traversal depth=1 by default (get_rel_map)
  - FR-009.4: Exports JSONL baseline; Parquet optional
  - FR-009.5: UI toggle + staleness badge
related_adrs: ["ADR-008","ADR-019","ADR-038"]
---

## Objective

Add optional GraphRAG using LlamaIndex PropertyGraphIndex. Compose via a RouterQueryEngine toolset (vector + graph) with safe fallbacks. Provide a UI toggle to enable graph build and routing; current default is ON (disable via `DOCMIND_ENABLE_GRAPHRAG=false`).

Update: Align with library-first, documented APIs only. Use `PropertyGraphIndex.as_retriever/as_query_engine` and export graph relations using `property_graph_store.get_rel_map(...)`. Compose router tools `[vector_query_engine, graph_query_engine(include_text=true, path_depth=1)]` with PydanticSingleSelector (OpenAI) else LLMSingleSelector. Persist via SnapshotManager (SPEC‑014) and show a staleness badge in Chat.

Seed Policy

- Prefer retriever-based seeds:
  - Graph: `PropertyGraphIndex.as_retriever(similarity_top_k=cap, path_depth=1)`
  - Else Vector: `VectorStoreIndex.as_retriever(similarity_top_k=cap)`
  - Else deterministic fallback
- Cap=32, deduplicate, stable tie-break (id asc). Seeds used in exports and can be surfaced in UI.

Relation Labels in Exports

- When the backend returns typed relations via `get_rel_map`, preserve the label in `relation`; otherwise fallback to `related`.

## Libraries and Imports

```python
from llama_index.core import PropertyGraphIndex
```

## File Operations

### CREATE / UPDATE

- `src/retrieval/graph_config.py`: traversal helpers using `get_rel_map`, portable exports to JSONL/Parquet. No index mutation; pure helpers and a small factory to build retriever/query_engine.
- `src/retrieval/router_factory.py`: build router with vector+graph tools, safe fallbacks
- `src/persistence/snapshot.py`: SnapshotManager (see SPEC‑014)

### UPDATE

- `src/pages/02_documents.py`: toggle for GraphRAG; optional PropertyGraphIndex build; export buttons; snapshot creation
- `src/pages/01_chat.py`: default router when graph present; staleness badge from manifest

## Acceptance Criteria

```gherkin
Feature: GraphRAG with Router and Persistence
  Scenario: Enable graph-aware retrieval
    Given GraphRAG is enabled in Settings
    And a PropertyGraphIndex is built
    When I query
    Then the router SHALL select between vector and graph tools (fallback to vector when graph missing)
    And graph exports SHALL be produced from get_rel_map() to JSONL and (optionally) Parquet

  Scenario: Snapshot and staleness
    Given a snapshot with manifest corpus_hash/config_hash
    And current hashes differ
    When I open Chat
    Then a staleness badge SHALL be visible
```

## References

- LlamaIndex Property Graph Guide: <https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide>
- Property Graph Examples: <https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_basic>
- Graph store API (`get`, `get_rel_map`, `save_networkx_graph`): see LlamaIndex examples (Context7 snippets)

## Changelog

- 1.1.0 (2025-09-09): Added router composition, SnapshotManager integration, staleness badge and acceptance criteria; library‑first update
