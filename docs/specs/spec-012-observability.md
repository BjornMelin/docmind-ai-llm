---
spec: SPEC-012
title: Observability: Structured Logging, Basic Metrics, Timing
version: 1.0.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - NFR-OBS-001: Log key events with structured dicts (jsonl).
  - NFR-OBS-002: Track timing for retrieval, rerank, generation.
related_adrs: ["ADR-001","ADR-010"]
---


## Objective

Add minimal structured logging and metrics across retrieval/rerank/compose steps. No external telemetry by default. Define canonical metric keys and units.

## File Operations

### UPDATE

- `src/utils/monitoring.py`: context manager `record_timing(name)` and `log_event(event: dict)`.

## Canonical Telemetry Keys

- retrieval.fusion_mode: rrf|dbsf (string)
- retrieval.prefetch_dense_limit, retrieval.prefetch_sparse_limit, retrieval.fused_limit (int)
- retrieval.rrf_k (int)
- retrieval.latency_ms (int)
- dedup.before, dedup.after, dedup.dropped (int)
- dedup.key (string)
- rerank.stage: text|visual|colpali (string)
- rerank.topk (int)
- rerank.latency_ms (int)
- rerank.timeout (bool)
- rerank.delta_mrr_at_10 (float)

All events SHALL be JSON lines; timestamps and correlation ids are recommended.

## Acceptance Criteria

```gherkin
Feature: Timing
  Scenario: Query round-trip
    When I ask a question
    Then logs SHALL include durations for retrieval and generation

Feature: Telemetry schema
  Scenario: Retrieval and Rerank events
    When a hybrid query and rerank execute
    Then logs SHALL include keys under retrieval.*, dedup.*, rerank.* as defined above
```
