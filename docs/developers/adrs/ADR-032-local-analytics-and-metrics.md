---
ADR: 032
Title: Local Analytics & Metrics (DuckDB)
Status: Proposed
Version: 1.0
Date: 2025-09-02
Supersedes:
Superseded-by:
Related: 030, 010, 012
Tags: analytics, metrics, duckdb
References:
- DuckDB
---

## Description

Optional local analytics DB (DuckDB) to record lightweight metrics. Disabled by default; separate from cache and ops stores.

## Context

Users may want local insights without external services.

## Decision Drivers

- Local‑first; minimal footprint

## Alternatives

- External services — rejected by default
- No analytics — acceptable default

## Decision

Provide simple schema and retention; write metrics only when enabled.

### Decision Framework

| Option                 | Local (35%) | Simplicity (35%) | Value (20%) | Maintain (10%) | Total | Decision      |
| ---------------------- | ----------- | ---------------- | ----------- | -------------- | ----- | ------------- |
| Separate DuckDB (Sel.) | 10          | 9                | 8           | 8              | 9.1   | ✅ Selected    |
| Couple with cache      | 8           | 6                | 8           | 7              | 7.4   | Rejected      |
| External service       | 2           | 4                | 9           | 6              | 4.7   | Rejected      |

## High-Level Architecture

App → ingestion helpers → DuckDB analytics

## Consequences

### Positive Outcomes

- Useful local insights; decoupled from core stores

### Dependencies

- Python: `duckdb`

## Changelog

- 1.0 (2025‑09‑02): Proposed optional analytics DB
