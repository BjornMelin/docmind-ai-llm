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

Add minimal structured logging and metrics across retrieval/rerank/compose steps. No external telemetry by default.

## File Operations

### UPDATE

- `src/utils/monitoring.py`: context manager `record_timing(name)` and `log_event(event: dict)`.

## Acceptance Criteria

```gherkin
Feature: Timing
  Scenario: Query round-trip
    When I ask a question
    Then logs SHALL include durations for retrieval and generation
```
