---
ADR: 019
Title: Optional GraphRAG via LlamaIndex PropertyGraphIndex
Status: Implemented
Version: 3.1
Date: 2025-08-19
Supersedes:
Superseded-by:
Related: 002, 003, 004, 031
Tags: graphrag, retrieval, relationships
References:
- LlamaIndex PropertyGraphIndex
---

## Description

Add an optional GraphRAG module using LlamaIndex’s PropertyGraphIndex with in‑memory store. No extra services; integrates with existing embedding/storage.

## Context

Multi‑hop and relationship queries benefit from lightweight graph structure. Keep optional and local‑only.

## Decision Drivers

- Library‑first; zero new infra
- Complement adaptive retrieval for entity/relationship use‑cases

## Alternatives

- Always‑on graph layer — unnecessary overhead
- Custom graph code — higher maintenance

### Decision Framework

| Option           | Capability (40%) | Simplicity (40%) | Ops (20%) | Total | Decision |
| ---------------- | ---------------- | ---------------- | --------- | ----- | -------- |
| PropertyGraphIdx | 8                | 9                | 10        | 8.8   | ✅ Sel.  |

## Decision

Expose GraphRAG behind a feature flag; use built‑in PropertyGraphIndex and simple store.

## High-Level Architecture

Docs → entities/relations → in‑mem graph → graph/hybrid retrieval

## Design

### Implementation Details

```python
# skeleton
def build_graph(docs):
    # extract entities/relations and populate PropertyGraphIndex
    return None
```

### Configuration

```env
DOCMIND_GRAPHRAG__ENABLED=false
```

## Testing

- Evaluate graph‑assisted queries vs baseline (ADR‑012)

```python
def test_graphrag_disabled_by_default(settings):
    assert settings.graphrag.enabled is False
```

## Consequences

### Positive Outcomes

- Better answers for relationship queries

### Negative Consequences / Trade-offs

- Additional preprocessing time when enabled

### Dependencies

- Python: `llama-index` (pinned)

## Changelog

- 3.1 (2025‑09‑04): Standardized to template; added requirements/config/tests

- 3.0 (2025‑08‑19): Implemented optional GraphRAG
