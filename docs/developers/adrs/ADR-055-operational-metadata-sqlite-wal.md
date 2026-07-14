---
ADR: 055
Title: Operational Metadata Store (SQLite WAL)
Status: Superseded
Version: 2.0
Date: 2026-07-13
Supersedes:
Superseded-by: 030, 031, 058
Related: 030, 031, 032, 033, 052, 058
Tags: persistence, sqlite, wal, jobs, offline-first
---

## Decision

The proposed shared operational SQLite database was not implemented. DocMind
keeps persistence at each subsystem's existing owner instead of maintaining an
unused cross-cutting database configuration:

- agent and chat state use their dedicated SQLite-backed stores
- ingestion cache and analytics use their existing DuckDB boundaries
- snapshots use their manifest and storage contract
- background jobs remain process-local

The v2 configuration hard cut removes the proposed database path, WAL toggle,
and startup directory creation. A durable job store requires a new accepted ADR
with a real consumer and migration contract.

## Consequences

- Operators no longer see settings that imply an unavailable operational store.
- Existing chat, agent-state, cache, snapshot, and analytics persistence remains
  unchanged.
- This ADR is historical context and does not authorize implementation.
