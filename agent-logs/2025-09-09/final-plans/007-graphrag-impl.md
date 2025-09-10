# SPEC-006 — GraphRAG Implementation Details

Date: 2025-09-09

## Purpose

Enhance the existing PropertyGraphIndex configuration by adding optional exports, a UI ingestion toggle, and basic counters to aid observability. Exports are implemented as Parquet (PyArrow) and JSONL fallback helpers; wiring is triggered from the ingestion adapter when the checkbox is enabled.

## Prerequisites

- `src/retrieval/graph_config.py` present
- PyArrow installed (already in repo)
- Optional: NetworkX for GraphML exports (treat as optional dependency)

## Files to Update (Checklist)

- [x] `src/retrieval/graph_config.py` — traversal helpers via `get_rel_map`, export helpers; label-preserving JSONL; Parquet optional
- [x] `src/retrieval/router_factory.py` — build RouterQueryEngine with vector+graph tools (fallback to vector); depth default=1
- [x] `src/persistence/snapshot.py` — SnapshotManager (atomic snapshots + manifest hashing + lock; relpath hashing)
- [x] `src/pages/02_documents.py` — “Build GraphRAG (beta)” toggle, snapshot creation, export buttons
- [x] `src/ui/ingest_adapter.py` — builds graph and exports when enabled (implemented)

## Export Helpers

1) GraphML Export (optional networkx)

- Function: `export_graph_graphml(index, path)`
- Try to import `networkx` and build graph from property graph store
- If networkx not available, fall back to JSONL of triplets

2) Parquet Export (PyArrow)

- Function: `export_graph_parquet(index, path)`
- Collect triplets `(head, relation, tail, score?)` into an Arrow table and write Parquet files under `data/graph/`

## Counters and Caps

- Track total triplets extracted per document and overall; log counts
- Add a cap knob (e.g., `max_triplets_per_chunk`) to limit extraction for large corpora

## UI Toggle and Flow

- Documents page adds a checkbox `Enable GraphRAG`
- Ingestion adapter checks the flag and builds a `PropertyGraphIndex` post-ingest for the new docs, then writes exports under `data/graph/` (`graph.parquet` and `graph.jsonl`).

## Paths and Outputs

- Default output directory: `settings.data_dir / "graph"`
- Files:
  - `graph.parquet` (triplets)
  - `graph.graphml` (when networkx available) or `graph.jsonl` fallback

## Acceptance Criteria (Status)

- [x] When enabled, ingestion produces JSONL (baseline) and, when PyArrow is available, Parquet exports
- [x] Exports include preserved relation labels when available (fallback `related`)

Gherkin:

```gherkin
Feature: GraphRAG exports and Router
  Scenario: Enable GraphRAG during ingestion
    Given the Documents page with Enable GraphRAG checked
    When I ingest documents
    Then graph exports are written under data/graph/
    And a non-empty graph representation is produced
  Scenario: Router toolset
    Given a built PropertyGraphIndex
    When I ask a question
    Then the router SHALL include vector and graph tools and fall back to vector when graph is missing
```

## Testing and Notes

- Integration test builds a tiny graph (two docs with a simple relationship) and asserts export files exist and are non-empty
- Keep networkx optional to avoid heavy dependency footprint; JSONL fallback ensures portability

## Imports and Libraries

- Optional: `networkx` for GraphML; if absent, write JSONL
- PyArrow for Parquet output
- LlamaIndex property graph APIs already in repo

## Cross-Links

- UI ingestion toggle: 003-ui-multipage-impl.md
- Code snippets: add simple export helpers (see 011-code-snippets.md Section 14)

## No Backwards Compatibility

- Remove any previous GraphRAG export stubs or scripts that conflict with this implementation. Update imports to point to the new helpers in `src/retrieval/graph_config.py`.
