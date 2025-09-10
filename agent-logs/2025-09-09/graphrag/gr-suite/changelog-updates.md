# CHANGELOG Additions (Phase 2)

## GraphRAG

- Add SnapshotManager for atomic, versioned snapshots with manifest.json (corpus/config hashing; staleness badge in Chat).

## Router

- Wire RouterQueryEngine (vector + graph) with safe fallbacks; expose via settings override.

## Graph helpers

- Library-first traversal using property_graph_store.get_rel_map; exports in JSONL (baseline) and Parquet (optional).

## UI

- Documents page toggle "Build GraphRAG (beta)", export buttons (seed cap=32), snapshot path display, Snapshot Utilities with “Rebuild GraphRAG Snapshot” button; Chat page defaults to router when graph present and shows clearer staleness copy.

## Tests

- Unit (router override, graph helpers, persistence), integration (exports, ingest→router), E2E smoke for Chat with router.

## Docs

- Final research and planning suite under agent-logs/2025-09-09/graphrag/gr-suite/; ADR‑019 and FR‑009 updates.
- New ADR‑038: GraphRAG router + SnapshotManager; accepted post‑merge.
- New SPEC‑014: Index Persistence Snapshots (SnapshotManager) with manifest hashing + lock + staleness UI.
- Amended ADRs: 003 (router composition), 013 (UI toggle/staleness/exports), 016 (session keys), 019 (router+persistence), 022 (graph export schema), 024 (GraphRAG flags), 031 (SnapshotManager), 033 (snapshot retention), 034 (corpus/config hash use).
- Revised SPECs: 006 (router + persistence + ACs; library‑first get_rel_map), 004 (router interop note), 002 (optional GraphRAG build note).
- Requirements: FR‑009 expanded (009.1–009.6); status set to Planned (Phase‑2); AC‑FR‑009 added.
- Traceability: FR‑009 row updated (ADRs, code, tests); Final plans 001/002/007/009 updated with new tasks and cross‑links.
