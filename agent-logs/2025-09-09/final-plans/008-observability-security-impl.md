# SPEC-012 + ADR-024/038/039 — Observability and Security Implementation

Date: 2025-09-09

## Purpose

Consolidate telemetry and security posture: structured JSONL metrics with required fields, optional analytics DB (DuckDB), endpoint allowlist/egress control, redacted logging, and AES-GCM image encryption.

## Observability

- Continue using `src/utils/telemetry.py` JSONL with sampling and rotation
- Required fields in retrieval events:
  - `retrieval.fusion_mode`, `retrieval.prefetch_dense_limit`, `retrieval.prefetch_sparse_limit`, `retrieval.fused_limit`
  - `retrieval.return_count`, `retrieval.latency_ms`
  - `dedup.before`, `dedup.after`, `dedup.dropped`, `dedup.key`
  - `retrieval.sparse_fallback` (when sparse prefetch skipped)
- Reranking enrichments:
  - Chosen visual path: `siglip_only` or `siglip_then_colpali`
  - Time budgets and whether they were exceeded (fail-open)
- Coordinator (best-effort): `coordination_overhead_ms`

## Security

- Endpoint allowlist with egress OFF by default
  - Validate outbound endpoints (e.g., LLM servers) against allowlist
  - Default to local-only (`localhost`, `127.0.0.1`)
- Redacted logging
  - Ensure telemetry events do not contain sensitive payloads; keep metrics only
- AES-GCM image encryption
  - Already implemented for page images; verify key management and defaults
- Restricted HTTP clients
  - Where HTTP clients are used, set `trust_env=False` by default and validate endpoints

## Acceptance Criteria

- Telemetry events include required fields; no sensitive content
- Reranking logs selected path and timeouts used
- Disallowed endpoints are blocked when egress is disabled
- Image encryption defaults protect page images at rest

Gherkin:

```gherkin
Feature: Observability and security posture
  Scenario: Telemetry completeness
    Given a hybrid retrieval query
    When it completes
    Then required fields are present in JSONL logs

  Scenario: Endpoint allowlist enforcement
    Given egress_enabled=false
    When a request targets a non-local endpoint
    Then the connection is rejected with an actionable message
```

## Testing and Notes

- Unit: verify telemetry JSONL contains the expected keys on a mocked query
- Security: attempt to set a remote URL when egress is disabled; assert rejection

## Imports and Libraries

- Telemetry: `json`, `datetime`, `pathlib.Path`
- Security helpers: `urllib.parse.urlparse`, `httpx` (if used)

## Cross-Links

- Retrieval and reranking telemetry points: existing modules emit JSONL (reference 011-code-snippets Section 15 for example events)
- Security allowlist notes and test ideas: see 005-final-spec-adr-plans.md (ADR-038 intent)

## No Backwards Compatibility

- Remove any legacy logging formats or PII-prone logs that conflict with JSONL metrics. Ensure all modules route observability through `src/utils/telemetry.py` and/or ADR-032 analytics.
