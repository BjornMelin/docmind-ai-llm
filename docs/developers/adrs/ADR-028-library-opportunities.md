---
ADR: 028
Title: Library Opportunities (Consolidation and Upgrades)
Status: Accepted
Version: 1.0
Date: 2025-08-19
Supersedes:
Superseded-by:
Related: 002, 003, 024
Tags: libraries, upgrades
References:
- Release notes (llama-index, qdrant-client)
---

## Description

Prefer modern built‑ins (e.g., LlamaIndex plugins, FlagEmbedding) over custom code; periodically review releases for simplifications.

## Context

Several areas can shrink with library features (embedding, retrieval helpers, cache adapters).

## Decision Drivers

- Reduce code; fewer bugs
- Pin tested versions

## Decision

Adopt “library‑first” reviews each milestone; remove custom layers when equivalent features exist.

## Decision Framework

| Candidate Area     | Library Feature            | Simplicity (40%) | Risk (20%) | Gain (40%) | Total | Decision |
|--------------------|----------------------------|------------------|------------|------------|-------|----------|
| Embeddings         | FlagEmbedding BGEM3        | 9                | 8          | 9          | 8.9   | ✅ Sel.  |
| Retrieval helpers  | LlamaIndex retrievers      | 9                | 8          | 9          | 8.9   | ✅ Sel.  |
| Cache adapters     | IngestionCache + DuckDB     | 9                | 9          | 9          | 9.0   | ✅ Sel.  |

## Consequences

### Positive Outcomes

- Smaller codebase; faster updates

### Dependencies

- pin: `llama-index>=0.10`, `FlagEmbedding>=1.2`, `qdrant-client>=1.6`

## Changelog

- 1.0 (2025‑08‑19): Accepted; library‑first policy recorded
