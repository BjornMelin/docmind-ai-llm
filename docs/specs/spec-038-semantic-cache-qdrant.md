---
spec: SPEC-038
title: Semantic Response Cache (Qdrant-backed, Guardrailed)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - FR-026: Optional semantic response caching for repeated/near-duplicate requests.
  - NFR-SEC-001: Offline-first; remote endpoints gated.
  - NFR-MAINT-003: No placeholder APIs; docs/specs/RTM match code.
related_adrs: ["ADR-035", "ADR-024", "ADR-031", "ADR-010", "ADR-004"]
---

## Goals

1. Add an optional response cache that can return answers for:
   - **exact** repeats (prompt_key match)
   - **near duplicates** (semantic similarity above threshold)
2. Preserve correctness with strict invalidation:
   - corpus/config hash changes must invalidate cache entries
   - model/template/params changes must not cross-hit
3. Keep offline-first posture and deterministic tests (no network in CI).

## Non-goals

- Provider-managed caching as the primary mechanism (Anthropic prompt caching is provider-specific).
- Introducing external services (Redis/LiteLLM proxy) by default.

## Design

### Storage Backend

Dedicated Qdrant collection (example: `docmind_semcache`) storing:

- vector: prompt embedding (dense, cosine distance)
- payload:
  - `prompt_key` (hash of canonicalized request payload; exact-match fast path)
  - `namespace`
  - `model_id`
  - `template_id`, `template_version`
  - `temperature`, `top_p` (or a normalized param signature)
  - `corpus_hash`, `config_hash`
  - `created_at`, `expires_at`
  - `response_ref` (either response text or a pointer to encrypted blob on disk)

> **Note on payload design**: Metadata fields (`model_id`, `template_id`, `corpus_hash`, `config_hash`, etc.) are required for strict invalidation and metadata filtering during cache lookups. No original prompt text is stored (see Security below). The `response_ref` field points to a local encrypted blob when encryption is enabled; the path itself does not contain PII. Observability events exclude sensitive fields (see Observability below).

### Read Path

1. Canonicalize request → `prompt_key`.
2. Exact match lookup:
   - Qdrant filter: `prompt_key == ...` and strict metadata filters.
3. Semantic lookup (only when allowed by config):
   - embed request prompt
   - Qdrant vector search + strict metadata filters
   - accept only if score >= `score_threshold`

### Write Path

- Only store responses when:
  - `settings.semantic_cache.enabled == true`
  - response bytes <= `max_response_bytes`
  - request type is allowed (optional allowlist)
- TTL enforced by `expires_at`.

### Invalidation

The cache must include:

- `corpus_hash` (from SnapshotManager hashing) and
- `config_hash` (from current ingestion/retrieval config)

Any mismatch prevents hits.

> **corpus_hash contract**: `corpus_hash` is computed over corpus file metadata (file paths, sizes, and modification times) via `SnapshotManager.compute_corpus_hash()`. It is recomputed after ingestion completes or during snapshot finalization. Changes to files, document re-indexing, or embedding model updates will change the hash and invalidate cache entries. See ADR-035 for the broader invalidation strategy.

### Configuration

Update `SemanticCacheConfig` in `src/config/settings.py`:

```python
class SemanticCacheConfig(BaseModel):
    enabled: bool = False
    provider: Literal["qdrant", "none"] = "qdrant"
    collection_name: str | None = None  # Override default collection name
    score_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    ttl_seconds: int = Field(default=1209600, ge=0)  # 14 days
    top_k: int = Field(default=5, ge=1)
    max_response_bytes: int = Field(default=24000, ge=0)
    namespace: str = "default"
    allow_semantic_for_templates: list[str] | None = None  # Optional allowlist
```

Fields:
- `provider`: Cache backend (`"qdrant"` or `"none"` when explicitly disabled)
- `collection_name`: Optional override for Qdrant collection name
- `allow_semantic_for_templates`: Optional allowlist of template IDs for semantic matching
- Existing fields preserved: `enabled`, `score_threshold`, `ttl_seconds`, `top_k`, `max_response_bytes`, `namespace`

### Observability

Emit local JSONL events via `log_jsonl()` (default log level: INFO):

- `semantic_cache_hit` with `{kind: exact|semantic, score, duration_ms, template_id}`
- `semantic_cache_miss` with `{duration_ms, template_id}`
- `semantic_cache_store` with `{duration_ms, bytes, template_id}` (DEBUG level)

**Safe-to-log fields**: `score`, `duration_ms`, `bytes`, `kind` are safe. `template_id` is safe as it does not contain PII (only identifier strings). Do not log `model_id`, `corpus_hash`, or response content in observability events. Conforms to ADR-047 (Safe Logging Policy) and project traceability guidance.

### Security

- **Vector storage only**: Store prompt embeddings (vectors) for semantic search; never store the original prompt text in the payload.
- **Response encryption**: Use optional AES-GCM encryption (existing helpers) for cached response payloads when handling sensitive content (PII, credentials, regulated data). Encrypt response blobs when stored outside trusted local disk; keep it optional for non-sensitive, performance-critical caches.
- **Metadata logging**: Never log secrets, API keys, or PII in cache metadata or observability events.

## Testing Strategy

### Unit

- Canonicalization → stable prompt_key.
- Exact-match hit/miss logic.
- Semantic hit requires score threshold.
- Invalidation by corpus/config hash prevents hits.

### Integration

- Use `QdrantClient(location=":memory:")` for deterministic offline tests.
- Verify cache integrates with LLM call path (MockLLM).

## Rollout / Migration

- Default off.
- Safe to ship incrementally; failures bypass cache.

## RTM Updates

Update `docs/specs/traceability.md`:

- Add row `FR-026` with code + tests once implemented.
- Implementation PRs must include RTM updates linking code + tests for FR-026.
