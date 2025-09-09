---
spec: SPEC-014
title: Index Persistence Snapshots (SnapshotManager)
version: 1.0.0
date: 2025-09-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - FR-009.2: SnapshotManager persistence + manifest + lock
  - FR-009.5: Staleness badge integration
related_adrs: ["ADR-038","ADR-031","ADR-034","ADR-019"]
---

## Objective

Define atomic, versioned snapshot persistence for indices and the property graph store. Provide a manifest with corpus/config hashes for staleness detection, and a lockfile to guarantee single‑writer safety.

## Filesystem Layout

```text
storage/
  _tmp-<uuid>/
    vector/           # StorageContext.persist output for vector index(es)
    graph/            # SimpleGraphStore JSON persisted file(s)
    manifest.json     # Snapshot manifest (see schema)
  <timestamp>/        # Finalized snapshot (atomic rename from _tmp)
```

## Manifest Schema

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

## Hashing Rules

- corpus_hash: stable SHA‑256 over sorted list of `(relative_path, size, mtime_ns)` for ingested files
- config_hash: stable SHA‑256 over canonical JSON of retrieval/chunking/embedding settings and toggles

## Atomicity & Locking

- Write all artifacts under `storage/_tmp-<uuid>` and `fsync` + atomic rename to `storage/<timestamp>`
- Create a `storage/.lock` file (e.g., `filelock`) around snapshot creation; timeout raises a friendly error in UI

## UI Integration

- Documents page: show snapshot path after ingest; provide rebuild action
- Chat page: read latest `manifest.json`, compute hashes, show staleness badge when mismatched; offer rebuild

## Acceptance Criteria

```gherkin
Feature: Atomic snapshots with manifest
  Scenario: Persist a new snapshot
    Given a vector index and a property graph store
    When I create a snapshot
    Then storage/<timestamp> SHALL exist with vector/, graph/, and manifest.json
    And the write SHALL occur under a lock and end with an atomic rename

  Scenario: Staleness detection
    Given a snapshot manifest with corpus_hash and config_hash
    And current corpus/config that differ
    When I open Chat
    Then a staleness badge SHALL be visible
```

## References

- ADR‑038 GraphRAG Persistence and Router Integration
- ADR‑031 Local‑first Persistence Architecture
- ADR‑034 Idempotent Indexing and Embedding Reuse

## Changelog

- 1.0.0 (2025-09-09): Initial draft spec
