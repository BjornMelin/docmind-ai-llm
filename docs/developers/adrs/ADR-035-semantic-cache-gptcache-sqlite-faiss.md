---
ADR: 035
Title: Application-Level Semantic Cache with GPTCache (SQLite + FAISS)
Status: Accepted
Version: 1.1.0
Date: 2025-09-03
Supersedes:
Superseded-by:
Related: 030, 031
Tags: cache, semantic, gptcache
References:
- GPTCache docs
---

## Description

Adopt an optional, in‑process semantic cache for prompt→response pairs using GPTCache with SQLite + FAISS. Feature‑flagged (Off by default).

## Context

Separates response caching from processing cache (ADR‑030). Keeps ops minimal and deterministic.

## Decision Drivers

- KISS; offline‑deterministic CI; safe‑by‑default

## Alternatives

- Qdrant‑backed semantic cache — heavier dep
- Proxy‑backed caches — adds services

### Decision Framework

| Option                    | Simplicity (35%) | Offline (25%) | Ops (20%) | Fit (20%) | Total | Decision      |
| ------------------------- | ---------------- | ------------- | --------- | --------- | ----- | ------------- |
| GPTCache (Sel.)           | 0.95             | 1.0           | 0.95      | 0.85      | 0.94  | ✅ Selected    |
| Qdrant semantic cache     | 0.70             | 0.9           | 0.70      | 0.90      | 0.79  | Rejected      |
| Proxy‑backed cache        | 0.55             | 0.3           | 0.40      | 0.80      | 0.51  | Rejected      |

## Decision

Use GPTCache behind a thin provider adapter with minimal parameters (e.g., score_threshold).

## High-Level Architecture

App → provider adapter → GPTCache (SQLite + FAISS)

## Design

### Implementation Details

```python
def get_semantic_cache():
    # return configured GPTCache instance when enabled
    return None
```

## Testing

- Deterministic cache hits/misses; threshold behavior

## Consequences

### Positive Outcomes

- Lower latency/cost on repeats; local only

### Dependencies

- Python: `gptcache`, `faiss-cpu`

## Changelog

- 1.1.0 (2025‑09‑03): Accepted optional semantic cache
