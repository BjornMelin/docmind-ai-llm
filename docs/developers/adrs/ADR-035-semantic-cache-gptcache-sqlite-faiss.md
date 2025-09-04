---
ADR: 035
Title: Application-Level Semantic Cache v1.1 with GPTCache (SQLite + FAISS)
Status: Accepted
Version: 1.1.0
Date: 2025-09-03
Supersedes:
Superseded-by:
Related: 004, 014, 024, 030, 031
Tags: cache, semantic, gptcache, sqlite, faiss, offline
References:
- [GPTCache — GitHub](https://github.com/zilliztech/GPTCache)
- [GPTCache — Docs](https://gptcache.readthedocs.io/en/latest/)
- [LiteLLM Proxy — Caching](https://docs.litellm.ai/docs/proxy/caching)
- [Qdrant — Payload/Filters](https://qdrant.tech/documentation/concepts/payload/)
---

## Description

Adopt an optional, in-process semantic cache for repeated prompt responses using GPTCache with SQLite storage and FAISS vectors. Feature-flagged (default Off) and exposed via a thin provider-agnostic adapter to keep call sites stable.

## Context

We want to reduce cost/latency for repeated prompts while preserving offline determinism and minimal ops. ADR‑030/031 cover processing cache; this ADR focuses on application-level semantic response caching and remains separate by design.

## Decision Drivers

- KISS & library-first; zero extra services
- Offline-deterministic tests (no network/GPU)
- Minimal config; safe-by-default
- Privacy: avoid storing raw prompts; namespace keys

## Alternatives

- A: Qdrant-backed semantic cache — Pros: strong filters; Cons: heavier dep/surface
- B: LiteLLM Proxy cache — Pros: mature; Cons: adds proxy service; violates offline CI
- C: GPTCache (Selected) — Pros: in-process, file-backed, deterministic tests; Cons: FAISS footprint

### Decision Framework

| Model / Option               | Simplicity (35%) | Offline CI (25%) | Ops Overhead (20%) | Library Fit (20%) | Total Score | Decision      |
| ---------------------------- | ---------------- | ---------------- | ------------------ | ----------------- | ----------- | ------------- |
| GPTCache (Selected)          | 9.5              | 10               | 9.5                | 8.5               | **9.4**     | ✅ Selected    |
| Qdrant-backed                | 7.0              | 9.0              | 7.0                | 9.0               | 7.9         | Rejected      |
| LiteLLM Proxy                | 5.5              | 3.0              | 4.0                | 8.0               | 5.1         | Rejected      |

## Decision

Adopt GPTCache (SQLite + FAISS) behind a thin adapter for v1.1.0. Default Off; enable via settings. Guard cacheability by temperature and size; store prompt hashes and namespaces for privacy and isolation. Fail-safe: bypass cache on errors.

## High-Level Architecture

```mermaid
graph TD
  A[Query Text] --> B[Embed]
  B --> C[SemanticCache Adapter]
  C -->|get| D[GPTCache (SQLite + FAISS)]
  D -->|hit| E[Return Cached]
  D -->|miss| F[LLM Generate]
  F -->|set| D
  F --> E
```

## Related Requirements

### Functional Requirements

- FR‑1: Optionally serve responses from a semantic cache on near-duplicate prompts
- FR‑2: Provide provider-agnostic adapter interface

### Non-Functional Requirements

- NFR‑1: Offline/local; deterministic tests
- NFR‑2: Minimal config; feature-flag default Off
- NFR‑3: Privacy-preserving (hash prompts; namespace keys)

### Performance Requirements

- PR‑1: `get()` latency under ~75ms on consumer laptops

### Integration Requirements

- IR‑1: Add `settings.semantic_cache` block (ADR‑024)
- IR‑2: Integrate at pre-generation stage (read→compute→write)

## Design

### Architecture Overview

- Adapter `ISemanticCache(get/set/expire)` in `src/utils/semantic_cache.py`
- Provider `GPTCacheSemanticCache` configured with SQLite + FAISS; TTL enforced on read
- Namespacing for env/model/template/settings-hash + prompt_hash

### Implementation Details

In `src/utils/semantic_cache.py` (illustrative):

```python
from typing import Protocol, Callable
from hashlib import sha256
from loguru import logger

class ISemanticCache(Protocol):
    def get(self, prompt: str, meta: dict, embedding: list[float] | None = None) -> str | None: ...
    def set(self, prompt: str, response: str, meta: dict, embedding: list[float] | None = None) -> None: ...
```

### Configuration

```env
DOCMIND_SEMANTIC_CACHE__ENABLED=false
DOCMIND_SEMANTIC_CACHE__PROVIDER=gptcache
DOCMIND_SEMANTIC_CACHE__SCORE_THRESHOLD=0.85
DOCMIND_SEMANTIC_CACHE__TTL_SECONDS=1209600
DOCMIND_SEMANTIC_CACHE__TOP_K=5
DOCMIND_SEMANTIC_CACHE__MAX_RESPONSE_BYTES=24000
DOCMIND_SEMANTIC_CACHE__NAMESPACE=default
```

## Testing

```python
def test_cache_hit_miss(mock_cache, deterministic_embed):
    # miss -> set -> hit sequence with thresholds and TTL
    pass
```

## Consequences

### Positive Outcomes

- Minimal, in-process semantic cache; offline deterministic
- Clear privacy posture; provider abstraction for future migration

### Negative Consequences / Trade-offs

- FAISS dependency footprint; read-time TTL vs scheduled cleanup

### Ongoing Maintenance & Considerations

- Tune `score_threshold` after observing hit quality
- Consider lightweight cleanup to trim expired rows

### Dependencies

- Python: `gptcache`, `faiss-cpu`

## Changelog

- **1.1.0 (2025-09-03)**: Initial accepted version.
