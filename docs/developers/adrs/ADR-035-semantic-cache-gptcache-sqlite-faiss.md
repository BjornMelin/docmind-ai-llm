---
ADR: 035
Title: Application-Level Semantic Cache with GPTCache (SQLite + FAISS)
Status: Accepted
Version: 1.1.1
Date: 2025-09-03
Supersedes:
Superseded-by:
Related: 030, 031
Tags: cache, semantic, gptcache
References:
- [GPTCache — Documentation](https://gptcache.readthedocs.io/)
---

## Description

Adopt an optional, in‑process semantic cache for prompt→response pairs using GPTCache with SQLite + FAISS. Feature‑flagged (Off by default).

## Context

Separates response caching from processing cache (ADR‑030). Keeps ops minimal and deterministic.

## Decision Drivers

- KISS; offline‑deterministic CI; safe‑by‑default

## Alternatives

- Qdrant‑backed semantic cache — heavier dep
- LiteLLM Proxy semantic cache (redis/qdrant via proxy) — adds services

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

### Architecture Overview

- Adapter encapsulates GPTCache; callers see a simple get/set API

## Related Requirements

### Functional Requirements

- FR‑1: Cache prompt→response pairs; configurable threshold

### Non-Functional Requirements

- NFR‑1: Local‑only; deterministic behavior in CI

### Performance Requirements

- PR‑1: Lookups ≤5ms; inserts ≤20ms typical

### Integration Requirements

- IR‑1: Feature flag; adapter hides GPTCache internals

## Design

### Implementation Details

```python
from gptcache import cache
from gptcache.manager import get_data_manager
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

def get_semantic_cache(db_path: str = "./data/gptcache.sqlite"):
    data_manager = get_data_manager("sqlite,faiss", data_dir=db_path)
    cache.init(
        embedding_func=Onnx(),
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
    )
    return cache
```

### Configuration

```env
DOCMIND_CACHE__SEMANTIC_ENABLED=false
DOCMIND_CACHE__SEMANTIC_PATH=./data/gptcache.sqlite
```

## Testing

- Deterministic cache hits/misses; threshold behavior

## Consequences

### Positive Outcomes

- Lower latency/cost on repeats; local only

### Negative Consequences / Trade-offs

- Requires careful threshold tuning to avoid stale hits

### Ongoing Maintenance & Considerations

- Monitor cache size and hit rate; clear/rehydrate cache on major model changes

### Dependencies

- Python: `gptcache`, `faiss-cpu`

## Changelog

- 1.1.1 (2025‑09‑04): Standardized to template; added decision framework
- 1.1.0 (2025‑09‑03): Accepted optional semantic cache
