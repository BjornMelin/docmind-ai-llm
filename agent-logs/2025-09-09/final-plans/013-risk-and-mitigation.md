# Risks, Performance Budgets, and Mitigation

Date: 2025-09-09

## Technical Risks

| Risk | Probability | Impact | Mitigation |
|---|---:|---:|---|
| Streamlit API drift | Low | Medium | Pin version, use supported APIs only |
| DuckDB contention | Low | Medium | Background queue, short-lived connections, small statements |
| BEIR dataset prep | Medium | Medium | Document acquisition; offer `--max_docs`; reuse collection via `--collection` |
| RAGAS evaluator LLM | Medium | Medium | Prefer metrics that work with a local evaluator wrapper; document limitations |
| Model download failures | Medium | High | CLI retries; print guidance; allow `--add` pairs for alternates |
| Sparse encoder unavailability | Medium | Low | Dense-only fallback; log `retrieval.sparse_fallback` |
| Optional GraphML dependency | Medium | Low | JSONL fallback |

## Implementation Risks

| Risk | Probability | Impact | Mitigation |
|---|---:|---:|---|
| Branch conflicts | Medium | Medium | Small focused branches; frequent rebases |
| Test flakiness | Medium | Medium | Mock network/heavy deps; deterministic seeds |
| Documentation drift | Medium | Low | Centralize code snippets; cross-link plans and RTM |

## Operational Risks

| Risk | Probability | Impact | Mitigation |
|---|---:|---:|---|
| Egress accidentally enabled | Low | High | Default OFF; allowlist enforced; tests |
| Telemetry PII leakage | Low | High | Redaction policy; keep metrics-only fields |
| Cache/DB growth | Medium | Medium | Retention/pruning; document clean-up |

## Performance Budgets

- UI page switch: < 2s
- Streamed first-chunk: ~100ms (best-effort)
- Analytics query: < 100ms
- BEIR (100 queries): < 30s
- RAGAS (50 cases): < 60s

## Contingencies

- If BEIR download fails: bundle tiny subsets, document manual placement under `data/beir/<dataset>`
- If HF Hub rate limits: instruct users to pre-download at off-peak times; rely on cache + offline flags
- If analytics degrades performance: disable analytics; writes are best-effort and can drop when necessary

## Rollbacks and Monitoring

- If a new feature causes instability, feature-flag it off via environment (e.g., disable analytics, disable GraphRAG toggle)
- Keep telemetry sampling enabled (low rate) to monitor regressions without overhead
- Maintain a simple “health” system test to exercise ingestion → retrieval → rerank paths after changes
