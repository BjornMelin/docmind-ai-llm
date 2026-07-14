---
spec: SPEC-006
title: GraphRAG: PropertyGraphIndex with Router and Library‑First Helpers
version: 1.3.0
date: 2026-07-14
owners: ["ai-arch"]
status: Accepted
related_requirements:
  - FR-009.1: Router engine wiring (vector+graph; fallback)
  - FR-009.3: Traversal depth=1 by default (get_rel_map)
  - FR-009.4: Exports JSONL baseline; Parquet optional
  - FR-009.5: UI toggle + staleness badge
related_adrs: ["ADR-008","ADR-019","ADR-038"]
---

## Objective

Add optional GraphRAG using LlamaIndex `PropertyGraphIndex`. Compose it in the
canonical `RouterQueryEngine` toolset and provide a UI toggle for graph build and
routing. GraphRAG ingestion defaults to off. Router inclusion has no second
feature flag: a supplied, healthy property graph index is the availability gate.

Use `PropertyGraphIndex.as_retriever/as_query_engine` and export graph relations
through `property_graph_store.get_rel_map(...)`. Compose the graph query engine
into the canonical native LlamaIndex router when healthy. Persist through
SnapshotManager (SPEC-014) with tri-file manifests and `graph_exports` metadata,
then surface export details and staleness in Chat. Router helpers rehydrate
indices directly from snapshot manifests and avoid bespoke synonym retrievers.

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
- Developer guide (direct LlamaIndex runtime and health checks): `docs/developers/guides/graphrag.md`

## Changelog

- 1.2.0 (2025-09-16): Documented `graph_exports` metadata, graph export telemetry fields, and snapshot `graph/` packaging.
- 1.1.0 (2025-09-09): Added router composition, SnapshotManager integration, staleness badge and acceptance criteria; library‑first update

## Graph Exports & Manifest Integration

- Graph exports SHALL be generated inside the active snapshot workspace under
  `graph/graph_export-snapshot-YYYYMMDDTHHMMSSZ.<ext>` before manifest hashing.
- Snapshot manifests (`manifest.meta.json`) MUST include export metadata:
  basename-only `filename`, `format`, `created_at`, `seed_count`, `size_bytes`,
  `duration_ms`, and `sha256`. The enclosing `graph/` directory is implicit.
- UI components and CLI tooling MUST read export information from `manifest.graph_exports`; no direct filesystem scans outside the manifest are allowed.
- Telemetry: each snapshot or manual export MUST emit a local
  `export_performed` event. It carries the same identity measurements with
  telemetry-specific names such as `dest_basename`, `export_type`, and `context`;
  it is not a byte-for-byte copy of the manifest entry.

## Observability

- Graph export helpers emit OpenTelemetry `graph_export.<format>` spans with
  `graph.export.adapter_name`, `graph.export.format`, `graph.export.depth`, and
  `graph.export.seed_count`. Completion adds an `export_performed` span event with
  `file.name` and `size.bytes`.
- Router construction emits an OpenTelemetry `router_selected` event with
  `tool.count` and `tool.names`. DocMind does not emit a per-query route or
  traversal-depth JSONL event.

## Exports & Seeds

- JSONL export is REQUIRED as the baseline; Parquet export is OPTIONAL and only available when `pyarrow` is installed.
- Seed selection MUST be deterministic, de‑duplicated, and capped at 32 items.
  - Seeds MAY be derived from ingested documents (e.g., unique `page_id`s) or from a top‑K pass via the retriever.
- Export file naming MUST include a Zulu timestamp and format suffix. Snapshot
  exports use a `graph_export-snapshot-` prefix and manual exports use
  `graph_export-manual-`. Successful exports emit telemetry events capturing
  export type, destination basename, seed count, byte count, duration, and
  context (`manual` or `snapshot`).
