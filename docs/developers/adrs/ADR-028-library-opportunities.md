---
ADR: 028
Title: Library Opportunities (Consolidation and Upgrades)
Status: Accepted
Version: 1.4
Date: 2026-07-11
Supersedes:
Superseded-by:
Related: 002, 003, 024
Tags: libraries, upgrades
References:
- [LlamaIndex — Releases](https://github.com/run-llama/llama_index/releases)
- [qdrant-client — PyPI](https://pypi.org/project/qdrant-client/)
---

## Description

Prefer modern library integrations over custom model wrappers; periodically review releases for simplifications.

## Context

Several areas can shrink with library features (embedding, retrieval helpers, cache adapters).

## Decision Drivers

- Reduce code; fewer bugs
- Pin tested versions

## Decision

Adopt “library‑first” reviews each milestone; remove custom layers when equivalent features exist.

## Decision Framework

| Candidate Area | Library Feature | Simplicity (40%) | Risk (20%) | Gain (40%) | Total | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| Dense embeddings | LlamaIndex Hugging Face adapter with BGE-M3 | 9 | 9 | 9 | 9.0 | Selected |
| Retrieval helpers | LlamaIndex retrievers | 9 | 8 | 9 | 8.9 | Selected |
| Cache adapters | IngestionCache + DuckDB | 9 | 9 | 9 | 9.0 | Selected |

## Consequences

### Positive Outcomes

- Smaller codebase; faster updates

### Negative Consequences / Trade-offs

- Time spent evaluating may delay feature work

### Ongoing Maintenance & Considerations

- Track release notes for targeted upgrades; prefer minor releases

## Testing

Not applicable — exploration ADR; track adoption via follow‑up PRs and benchmarks.

## High-Level Architecture

- Periodic review → candidates list → decision log → incremental adoption

## Related Requirements

### Functional Requirements

- FR‑1: Capture candidate upgrades per milestone

### Non-Functional Requirements

- NFR‑1: Keep evaluation lightweight; prototype before committing

### Performance Requirements

- PR‑1: Measure perf against baseline before adopting

### Integration Requirements

- IR‑1: Record decisions in ADRs; update references

## Design

### Architecture Overview

- Lightweight doc + small spike PRs

### Implementation Details

- Maintain shortlist; prefer official libs/features

### Configuration

- Not applicable

### Dependencies

- Pin direct dependencies: `llama-index-core>=0.14.21,<0.15.0`, selected LlamaIndex integrations, `sentence-transformers>=5.2.0,<6.0.0`, and `qdrant-client>=1.15.1,<2.0.0`.

## Changelog

- 1.4 (2026-07-11): Removed the unused parallel FlagEmbedding stack and selected the existing LlamaIndex Hugging Face adapter as the single dense embedding owner.

- 1.3 (2026-07-10): Replace the removed LlamaIndex meta-package with the direct core and selected-integration contract.

- 1.2 (2025‑09‑12): Acknowledged consolidation of reranking stack per SPEC‑005/ADR‑037: removed legacy adapters, unified device policy and SigLIP loader, added encrypted image helper; library‑first replacements now canonical.

- 1.1 (2025‑09‑04): Standardized to template; added decision framework

- 1.0 (2025‑08‑19): Accepted; library‑first policy recorded
