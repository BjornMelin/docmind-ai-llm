---
ADR: 031
Title: Local-First Persistence Architecture
Status: Accepted
Version: 1.1
Date: 2025-09-03
Supersedes:
Superseded-by:
Related: 030, 002, 003
Tags: persistence, qdrant, duckdb, sqlite
References:
- Qdrant client
- LlamaIndex vector stores
---

## Description

Separate persistence by concern: Qdrant for vectors, DuckDBKVStore for processing cache, and SQLite (optional) for operational metadata.

## Context

One store doesn’t fit all needs; keep each component minimal and local.

## Decision Drivers

- Local‑only; maintainable; clear boundaries

## Alternatives

- External services — rejected by default
- Custom cache layers — redundant

## Decision

Adopt Qdrant (vectors), IngestionCache+DuckDB (cache), SQLite (ops metadata).

## High-Level Architecture

App → Processor → {Qdrant, DuckDB cache, SQLite}

## Consequences

### Positive Outcomes

- Right tool per concern; minimal code

### Dependencies

- Python: `qdrant-client`, `llama-index`, `duckdb`

## Changelog

- 1.1 (2025‑09‑03): Accepted; boundaries finalized
