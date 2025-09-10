# Docs/ADRs/SPECs/Plans — Master Update Guide and Checklist (GraphRAG Phase‑2)

## Objective

- Provide a single, fool‑proof, step‑by‑step guide to update all affected documentation artifacts (ADRs, SPECs, requirements, traceability, and final plan docs) to align with the finalized GraphRAG Phase‑2 design: RouterQueryEngine wiring, library‑first Graph helpers, SnapshotManager persistence, UI toggles, exports, and tests.
- Ensure no conflicting, outdated, or incorrect info remains; add missing ADRs/SPECs/requirements/traceability entries; cross‑reference consistently.

## Scope of Impact

- ADRs: docs/developers/adrs
- SPECs: docs/specs
- Requirements: docs/specs/requirements.md
- Traceability: docs/specs/traceability.md
- Final Plans: agent-logs/2025-09-09/final-plans

## New/Amended Artifacts to Add

- New ADR: ADR-038-graphrag-persistence-and-router.md
  - Status: Proposed → Accepted when merged
  - Amends: ADR-019-optional-graphrag.md
  - Relates: ADR-003, ADR-013, ADR-016, ADR-022, ADR-024, ADR-031, ADR-033, ADR-034
  - Summary: Define GraphRAG library‑first approach; RouterQueryEngine with vector+graph tools and safe fallbacks; SnapshotManager for atomic snapshots + manifest hashes; UI defaults + exports.

- New SPEC: spec-014-index-persistence-snapshots.md
  - Describes SnapshotManager filesystem layout, manifest schema (corpus_hash, config_hash, created_at, versions), atomic rename, lockfile, and staleness badge behavior.

- Requirements additions/updates
  - FR-009 (GraphRAG PropertyGraphIndex): expand with sub‑requirements FR‑009.1 … FR‑009.6 (router, persistence, traversal, exports, UI, tests); update status to “Planned (Phase‑2)” or “In progress” when implementation starts.
  - Optional new requirement FR‑021 (Router selection/staleness UI): Only if you want a separate line item. Otherwise include under FR‑009.

- Traceability: Expand FR‑009 mapping to include new code/tests; link ADR‑038, SPEC‑006/009.

## Inventory (as of current repo)

- ADRs relevant to update:
  - ADR-003-adaptive-retrieval-pipeline.md
  - ADR-013-user-interface-architecture.md
  - ADR-016-ui-state-management.md
  - ADR-019-optional-graphrag.md
  - ADR-022-export-output-formatting.md
  - ADR-024-configuration-architecture.md
  - ADR-031-local-first-persistence-architecture.md
  - ADR-033-local-backup-and-retention.md
  - ADR-034-idempotent-indexing-and-embedding-reuse.md

- SPECs relevant to update:
  - spec-006-graphrag.md
  - spec-004-hybrid-retrieval.md (reference router composition)
  - spec-002-ingestion-pipeline.md (optional graph build path)

- Requirements/Traceability:
  - docs/specs/requirements.md — update FR-009
  - docs/specs/traceability.md — update FR-009 row

- Final Plans to cross‑align:
  - 001-graphrag-finalization.md
  - 002-decisions-adr-specs.md
  - 007-graphrag-impl.md
  - 009-task-checklists.md
  - 010-acceptance-criteria-and-tests.md
  - 012-rtm-updates.md

## Master Checklist (mark [x] when done)

1) Add ADR-038 (new)

- [x] Create docs/developers/adrs/ADR-038-graphrag-persistence-and-router.md with sections:
  - [x] Context: library‑first GraphRAG, need for router + persistence
  - [x] Decision: RouterQueryEngine (vector + graph), SnapshotManager with manifest hashing and lock, UI defaults & exports
  - [x] Consequences: fallback to vector, staleness badge, export formats, feature flags
  - [x] Status: Proposed
  - [x] Amends ADR‑019; Relates ADR‑003/013/016/022/024/031/033/034

### Suggested ADR‑038 content blocks

- **Context:**
  - “We adopt LlamaIndex PropertyGraphIndex using documented APIs (get, get_rel_map), composing router query engine tools for vector and graph, and persisting indices via SnapshotManager with atomic snapshots and manifest hashes (corpus_hash, config_hash).”
- **Decision:**
  - “Default when graph exists: RouterQueryEngine with PydanticSingleSelector (OpenAI) else LLMSingleSelector, tools=[vector_query_engine, graph_query_engine(include_text, path_depth=1)]. If graph absent/unhealthy: vector tool only.”
  - “SnapshotManager writes to storage/_tmp-<uuid> then atomically renames to storage/<timestamp>, writes manifest.json with corpus_hash (ingested file signatures) and config_hash (retrieval/chunking/embedding settings), uses a lockfile to ensure single writer.”
  - “UI: Documents page toggle ‘Build GraphRAG (beta)’; Chat defaults to router when graph present; show staleness badge when hashes mismatch; export JSONL (baseline) and Parquet (optional with pyarrow).”
- **Consequences:**
  - “Router is robust to missing graph; staleness is discoverable; exports portable; later Cypher integrations gated behind explicit ADR.”

2) Amend ADR-019-optional-graphrag.md

- [x] Add “Amended by ADR‑038” in header metadata
- [x] Update “Decision” to reference RouterQueryEngine + SnapshotManager as the accepted approach
- [x] Remove/adjust any text implying direct index mutation or undocumented graph store methods
- [x] Link to SPEC‑006 and SPEC‑014

3) Update ADR-003-adaptive-retrieval-pipeline.md

- [x] Add subsection “GraphRAG Router Composition” describing vector + graph tools and selector logic; note safe fallback
- [x] Cross‑link ADR‑038 and SPEC‑006

4) Update ADR-013-user-interface-architecture.md

- [x] Add GraphRAG UI controls: Documents toggle; Chat staleness badge; export buttons
- [x] Cross‑link ADR‑024 (config flags) and ADR‑038

5) Update ADR-016-ui-state-management.md

- [x] Document new session state entries: vector_index, pg_index, router_engine, snapshot_manifest
- [x] Note loading latest snapshot on Chat page init

6) Update ADR-022-export-output-formatting.md

- [x] Specify Graph exports: JSONL baseline (rel_map rows), Parquet optional (pyarrow) with schema (subject, relation, object, depth, path_id, source_ids)
- [x] Cross‑link SPEC‑006 and SPEC‑014

7) Update ADR-024-configuration-architecture.md

- [x] Add config flags: graphrag.enabled (bool), graphrag.subretrievers (bool), graphrag.default_path_depth (int)
- [x] Describe default behavior and overrides

8) Update ADR-031-local-first-persistence-architecture.md

- [x] Incorporate SnapshotManager pattern: atomic rename, manifest fields, lockfile
- [x] Clarify interaction with StorageContext.persist and SimpleGraphStore.persist

9) Update ADR-033-local-backup-and-retention.md

- [x] Add snapshot directory retention policy (e.g., N latest or TTL)
- [x] Note manifest includes created_at and versions for audit

10) Update ADR-034-idempotent-indexing-and-embedding-reuse.md

- [x] Reference corpus_hash/config_hash as staleness and idempotency keys
- [x] Define how they’re computed (file path/size/mtime; settings fingerprint)

11) Add SPEC-014-index-persistence-snapshots.md (new)

- [x] Create docs/specs/spec-014-index-persistence-snapshots.md covering:
  - [x] Filesystem layout
  - [x] Manifest JSON schema
  - [x] Hashing algorithms (stable ordering; SHA256)
  - [x] Atomic rename and locking constraints
  - [x] Staleness UI rules
  - [x] Links: ADR‑038/031/034; SPEC‑006

12) Update SPEC-006-graphrag.md

- [x] Replace internal calls with library‑first terminology (get_rel_map, as_query_engine/as_retriever)
- [x] Add router tool composition and default selector logic
- [x] Add exports and persistence integration points (SnapshotManager)
- [x] Update acceptance criteria bullets (see below)

13) Update SPEC-004-hybrid-retrieval.md

- [x] Add note: router can compose hybrid retrievers behind a feature flag; default off
- [x] Cross‑link ADR‑038 and SPEC‑006

14) Update SPEC-002-ingestion-pipeline.md

- [x] Add optional GraphRAG build step after core ingestion; mention cost and toggle

15) Update Requirements (docs/specs/requirements.md)

- [x] FR-009: expand with sub‑requirements and ACs:
  - FR‑009.1 Router engine wiring (vector+graph; fallback)
  - FR‑009.2 SnapshotManager persistence + manifest + lock
  - FR‑009.3 Traversal (get_rel_map depth=1 default; cap limits)
  - FR‑009.4 Exports JSONL baseline; Parquet optional (pyarrow)
  - FR‑009.5 UI controls & staleness badge
  - FR‑009.6 Tests (unit, integration, E2E smoke)
- [x] Update “Status” for FR‑009 to Planned/Phase‑2 (or In progress when implemented)
- [x] Add ADR/SPEC references: ADR‑038, SPEC‑006, SPEC‑014

### Suggested FR‑009 AC bullets to insert

- “RouterQueryEngine selects between vector_query_engine and graph_query_engine; if graph missing, routes to vector only.”
- “SnapshotManager produces storage/<timestamp> with manifest.json including corpus_hash and config_hash; lock prevents concurrent writes; atomic rename.”
- “Graph traversal uses property_graph_store.get_rel_map with default path_depth=1; breadth limits enforced.”
- “Export produces valid JSON Lines; Parquet emitted when pyarrow present and schema matches.”
- “Documents page shows Build GraphRAG (beta) toggle; Chat shows staleness badge if manifest hashes differ from current.”
- “Tests: unit (router override, helpers, snapshot), integration (exports, ingest→router), E2E smoke (router answers with sources).”

16) Update Traceability (docs/specs/traceability.md)

- [x] Update FR-009 row:
  - ADR(s): 019, 038
  - Code: src/retrieval/graph_config.py; src/retrieval/router_factory.py; src/persistence/snapshot.py; src/pages/01_chat.py; src/pages/02_documents.py
  - Tests: tests/unit/agents/test_settings_override_router.py; tests/unit/retrieval/test_graph_helpers.py; tests/unit/persistence/test_snapshot_manager.py; tests/integration/test_graphrag_exports.py; tests/integration/test_ingest_router_flow.py; tests/e2e/test_chat_graphrag_smoke.py
  - Verification: test
  - Status: Planned/Phase‑2 → update as you implement

17) Update Final Plans (agent-logs/2025-09-09/final-plans)

- [x] 001-graphrag-finalization.md: add a summary bullet linking ADR‑038/SPEC‑014
- [x] 002-decisions-adr-specs.md: include “Create ADR‑038” and “Create SPEC‑014” tasks
- [x] 007-graphrag-impl.md: include SnapshotManager + Router factory modules; UI toggle; staleness badge; exports
- [x] 009-task-checklists.md: add checkboxes mirroring this guide
- [x] 010-acceptance-criteria-and-tests.md: add FR‑009 AC bullets from above
- [x] 012-rtm-updates.md: include FR‑009 mapping updates

18) Conflict and Outdated Content Removal

- [x] Search ADRs/SPECs for any references to:
  - direct index mutation for graph helpers

19) Phase‑2 Implementation Polish

- [x] Update Chat staleness badge copy to direct users to Documents → “Rebuild GraphRAG Snapshot”
- [x] Cap export seeds to 32 to bound latency
- [x] Add Documents page expander “About snapshots” and rebuild button
  - undocumented store calls like get_nodes/get_edges
  - GraphRAG as a standalone path without router involvement
  - lack of persistence or global state assumptions
- [x] Replace with library‑first guidance and SnapshotManager references

19) Cross‑reference consistency sweep

- [x] Ensure each ADR/SPEC cites the others appropriately:
  - ADR‑038 cites SPEC‑006/014 and ADR‑019
  - SPEC‑006 cites ADR‑038
  - Requirements/RTM cite ADR‑038 and SPEC‑006/014

## Appendix A — Ready-to-Paste headers

### ADR‑038 front matter (YAML or first block):

```json
Title: GraphRAG Persistence and Router Integration
Status: Proposed
Amends: ADR-019-optional-graphrag.md
Relates: ADR-003, ADR-013, ADR-016, ADR-022, ADR-024, ADR-031, ADR-033, ADR-034
Date: 2025-09-09
```

### SPEC‑014 header:

```json
# spec-014: Index Persistence Snapshots
Status: Draft
Relates: ADR-038, ADR-031, ADR-034, SPEC-006
Date: 2025-09-09
```

## Appendix B — Manifest JSON schema (SPEC-014)

```json
{
  "index_id": "string",
  "graph_store_type": "string",
  "vector_store_type": "string",
  "corpus_hash": "sha256:...",
  "config_hash": "sha256:...",
  "created_at": "YYYY-MM-DDTHH:MM:SSZ",
  "versions": {
    "llama_index": "x.y.z",
    "app": "x.y.z"
  }
}
```

## Appendix C — Graph Export Row (JSONL)

```json
{
  "subject": "node_id_or_name",
  "relation": "REL_LABEL",
  "object": "node_id_or_name",
  "depth": 0|1|2,
  "path_id": "uuid",
  "source_ids": ["doc_id:chunk_id", ...]
}
```

## Notes

- Keep language and structure consistent with existing ADR/SPEC templates.
- If you prefer to keep ADR‑019 immutable, mark ADR‑038 as Supersedes ADR‑019 and set ADR‑019 status to “Superseded” in a minimal follow‑up edit.
