# ADR-035: Application-Level Semantic Cache v1.1 with GPTCache (SQLite + FAISS)

## Metadata

**Status:** Accepted  
**Version/Date:** v1.1.0 / 2025-09-03

## Title

Optional, local-first semantic cache via GPTCache (SQLite + FAISS), behind a thin provider adapter

## Description

Adopt a minimal, in-process semantic cache for repeated prompt responses using GPTCache with SQLite storage and FAISS vector index. The feature is gated by a settings flag (default Off) and exposed via a thin provider-agnostic adapter so we can add Qdrant or LiteLLM Proxy later without changing call sites.

## Context

We need an optional semantic cache to reduce latency/cost on repeated prompts, while preserving:

- Offline-deterministic CI (no network/GPU)
- KISS/DRY/YAGNI and library-first principles
- Privacy (no PII in cache; prompt hashing)
- Minimal operational burden (no new services)

Prior caching ADRs (ADR-030/ADR-031) cover document-processing cache via LlamaIndex IngestionCache (DuckDBKVStore). This ADR is strictly about application-level semantic response caching. We deliberately separate these cache types.

## Decision Drivers

- KISS & library-first; zero extra services
- Offline-deterministic CI; deterministic tests
- Minimal config surface; safe-by-default
- Privacy and safety on cache hits/misses

## Alternatives

- **A**: Qdrant-backed semantic cache (client local/in-memory)
  - Pros: robust payload filters; aligns with vector infra
  - Cons: higher integration surface; TTL cleanup by filter; heavier dep
- **B**: LiteLLM Proxy semantic cache (redis/qdrant via proxy)
  - Pros: mature features, headers, centralization
  - Cons: adds a proxy service; network hop; conflicts with offline CI
- **C**: GPTCache (SQLite + FAISS) [Selected]
  - Pros: fully in-process, file-backed, low ops, deterministic tests
  - Cons: FAISS dependency; fewer operational knobs than a DB

### Decision Framework

| Option                         | Simplicity (35%) | Offline CI (25%) | Ops Overhead (20%) | Library Fit (20%) | Total | Decision      |
| ------------------------------ | ---------------- | ---------------- | ------------------ | ----------------- | ----- | ------------- |
| **C: GPTCache (Selected)**     | 0.95             | 1.0              | 0.95               | 0.85              | **0.94** | ✅ Selected |
| A: Qdrant semantic cache       | 0.70             | 0.9              | 0.70               | 0.9               | 0.79  | Rejected      |
| B: LiteLLM Proxy semantic      | 0.55             | 0.3              | 0.40               | 0.8               | 0.51  | Rejected      |

## Decision

We will adopt **GPTCache (SQLite + FAISS)** behind a thin provider adapter for v1.1.0.

- Default Off; enable via `settings.semantic_cache.enabled=true` and `provider=gptcache`.
- Parameters (defaults; settings-configurable):
  - `score_threshold=0.85` (cosine on L2-normalized vectors)
  - `ttl_seconds=1209600` (14 days)
  - `top_k=5`
  - `max_response_bytes=24_000`
  - Cacheable only when `temperature <= 0.2` (stable-response guard)
- Privacy: store `prompt_hash`, not raw prompts; namespace cache keys by env/model/template/settings-hash.
- Fail-safe: on any error, bypass cache silently.

## High-Level Architecture

```mermaid
graph TD
  A[Query Text] --> B[Embed (same model as retrieval)]
  B --> C[SemanticCache Adapter]
  C -->|get| D[GPTCache (SQLite + FAISS)]
  D -->|hit->response| E[Return Cached]
  D -->|miss| F[LLM Generate]
  F -->|set| D
  F --> E
```

## Related Requirements

### Functional Requirements

- **FR-1**: Optionally serve responses from a semantic cache when a near-duplicate prompt is seen.
- **FR-2**: Provide a provider-agnostic adapter to swap implementations.

### Non-Functional Requirements

- **NFR-1**: Fully offline/local; CI deterministic.
- **NFR-2**: Minimal settings and ops footprint; feature-flag default Off.
- **NFR-3**: Privacy-preserving; do not store PII or raw prompts.

### Performance Requirements

- **PR-1**: Cache `get()` under ~75ms budget on consumer laptops.
- **PR-2**: Maintain hit quality via `score_threshold` and temperature policy.

### Integration Requirements

- **IR-1**: New Pydantic block `settings.semantic_cache` (see ADR-024).
- **IR-2**: Single integration point in pre-generation stage (read→compute→write).

## Related Decisions

- **ADR-030**: IngestionCache (DuckDBKVStore) — document-processing cache; orthogonal to this ADR.
- **ADR-031**: Local-first persistence; recognizes separation between processing cache (DuckDB) and application semantic cache.
- **ADR-024**: Unified settings; adds `SemanticCacheConfig` mapping.
- **ADR-014**: Testing & quality; offline-deterministic tests with temp SQLite and MockEmbedding.
- **ADR-004**: Local-first LLM strategy; aligns with fully offline design.

## Design

### Architecture Overview

- **Adapter interface**: `ISemanticCache(get/set/expire)` in `src/utils/semantic_cache.py`.
- **Provider**: `GPTCacheSemanticCache` configured with SQLite + FAISS; TTL enforced on read (lazy expiry).
- **Namespacing**: key includes env, llm_model, template_version, settings_hash + prompt_hash.
- **Fail-safe**: log warnings at debug; never block app path.

### Implementation Details

**In `src/utils/semantic_cache.py`:**

```python
from typing import Protocol, Callable
from hashlib import sha256
from time import time as now
from loguru import logger
from gptcache import cache as gptcache
from gptcache.manager import get_data_manager, CacheBase, VectorBase

class ISemanticCache(Protocol):
    def get(self, prompt: str, meta: dict, embedding: list[float] | None = None) -> str | None: ...
    def set(self, prompt: str, response: str, meta: dict, embedding: list[float] | None = None) -> None: ...
    def expire(self, now_ms: int | None = None) -> int: ...

class GPTCacheSemanticCache:
    def __init__(self, embed_fn: Callable[[str], list[float]], dim: int, cfg):
        dm = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=dim))
        gptcache.init(embedding_func=embed_fn, data_manager=dm)
        self.cfg = cfg

    def get(self, prompt, meta, embedding=None):
        # 1) short-circuit guards
        if not meta.get("cacheable", False):
            return None
        if meta.get("temperature", 1.0) > 0.2:
            return None
        # 2) lookup by semantic similarity; TTL lazy check is enforced on read payload
        # (store created_at in payload; drop if expired)
        try:
            # pseudo: retrieve top hit and validate score + TTL
            # return response_text or None
            ...
        except Exception as e:
            logger.warning("semantic cache get failed: {}", e)
            return None

    def set(self, prompt, response, meta, embedding=None):
        try:
            if len(response.encode("utf-8")) > self.cfg.max_response_bytes:
                return
            # upsert with created_at, namespace fields
            ...
        except Exception as e:
            logger.warning("semantic cache set failed: {}", e)
```

### Configuration

**In `src/config/settings.py`:**

```python
class SemanticCacheConfig(BaseModel):
    enabled: bool = Field(default=False)
    provider: str = Field(default="gptcache")
    score_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    ttl_seconds: int = Field(default=1209600, ge=60, le=2592000)
    top_k: int = Field(default=5, ge=1, le=20)
    max_response_bytes: int = Field(default=24000, ge=1024, le=1048576)
    namespace: str = Field(default="default")
```

**In `.env.example`:**

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

- **Unit**: hit/miss behavior by threshold; TTL lazy expiry; size guard; namespace isolation; feature flag off path.
- **Integration**: end-to-end get→miss→compute→set→hit with temp SQLite; deterministic embeddings.
- **E2E**: unchanged pipeline correctness when feature is Off; fast path when On and cache warmed.

## Consequences

### Positive Outcomes

- Minimal, in-process, offline semantic cache
- Deterministic tests without network/GPU
- Clear privacy posture (no raw prompt persistence)
- Provider abstraction allows later Qdrant/LiteLLM migration

### Negative Consequences / Trade-offs

- FAISS dependency footprint
- Read-time TTL enforcement vs scheduled cleanup (can add lightweight cleanup)

### Ongoing Maintenance & Considerations

- Revisit `score_threshold` after observing hit quality locally
- Optional scheduled cleanup to trim expired rows
- Evaluate Qdrant provider when cache size/needs grow (future ADR)

### Dependencies

- **Python**: `gptcache`, `faiss-cpu`

## References

- GPTCache: <https://github.com/zilliztech/GPTCache>
- GPTCache docs (usage, similarity evaluation): <https://gptcache.readthedocs.io/en/latest/>
- LiteLLM Proxy caching: <https://docs.litellm.ai/docs/proxy/caching>
- Qdrant payload/filters: <https://qdrant.tech/documentation/concepts/payload/>
- Final research plan: agent-logs/2025-09-02/processing/002_semantic_cache_and_reranker_ui_final_plan.md

## Changelog

- **1.1.0 (2025-09-03)**: Initial accepted version.
