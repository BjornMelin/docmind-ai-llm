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

### Configuration

Update `SemanticCacheConfig` in `src/config/settings.py`:

- add provider `qdrant`
- add `collection_name` (optional override)
- add `allow_semantic_for_templates: list[str]` (optional)
- keep existing:
  - `enabled`, `score_threshold`, `ttl_seconds`, `top_k`, `max_response_bytes`, `namespace`

### Observability

Emit JSONL events:

- `semantic_cache_hit` with `{kind: exact|semantic, score, duration_ms, template_id}`
- `semantic_cache_miss` with `{duration_ms, template_id}`
- `semantic_cache_store` with `{duration_ms, bytes, template_id}`

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
