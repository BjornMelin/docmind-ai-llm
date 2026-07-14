---
spec: SPEC-039
title: Operational Metadata Store (SQLite WAL)
version: 2.0.0
date: 2026-07-13
owners: ["ai-arch"]
status: Superseded
related_requirements:
  - FR-025: Background jobs record progress/cancellation safely.
  - NFR-REL-001: Recovery after restart without corruption.
related_adrs: ["ADR-030", "ADR-031", "ADR-055", "ADR-058"]
notes: "Superseded by the shipped, subsystem-owned persistence boundaries."
---

## Supersession Notice

The proposed shared operational SQLite database was never implemented and is
not part of the supported DocMind architecture. Its unused configuration and
startup side effect were removed in the v2 hard cut.

Persistence remains with the subsystem that owns each contract:

- chat and agent state use their dedicated SQLite-backed stores
- ingestion cache uses DuckDB KV
- snapshots use the snapshot store and manifest contract
- analytics uses DuckDB
- background job coordination remains process-local

A future durable job store requires a new accepted spec backed by a concrete
consumer. This document does not authorize implementation.
