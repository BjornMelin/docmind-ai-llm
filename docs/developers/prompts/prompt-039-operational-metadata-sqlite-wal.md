---
prompt: PROMPT-039
title: Operational Metadata Store (SQLite WAL)
status: Superseded
date: 2026-07-13
version: 2.0
related_adrs: ["ADR-055"]
related_specs: ["SPEC-039"]
---

## Do Not Execute

The shared operational SQLite store was never implemented and is not part of
the supported architecture. ADR-055 and SPEC-039 are superseded.

DocMind keeps persistence with each live subsystem owner:

- chat and agent state use their dedicated SQLite-backed stores
- ingestion cache uses DuckDB KV
- snapshots use the snapshot store and manifest contract
- analytics uses DuckDB
- background jobs remain process-local

Do not recreate the removed settings or startup side effect. A durable job
store requires a new accepted ADR, spec, concrete consumer, and migration
contract.
