---
spec: SPEC-010
title: Evaluation: BEIR/M-BEIR for IR; RAGAS for End-to-End
version: 1.0.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-EVAL-001: Provide offline scripts to compute recall@k, nDCG, MRR on BEIR/M-BEIR.
  - FR-EVAL-002: Provide RAGAS pipeline with dataset adapters.
  - NFR-PERF-004: Track latency and memory budgets per profile.
related_adrs: ["ADR-011"]
---


## Objective

Ship offline evaluation scripts for IR and E2E. Record results in a small leaderboard CSV for regressions.

## Libraries and Imports

```python
from beir.datasets.data_loader import GenericDataLoader  # or datasets from HF
from ragas import evaluate
```

## File Operations

### CREATE

- `tools/eval/run_beir.py`, `tools/eval/run_ragas.py`.
- `data/eval/README.md` with dataset instructions.

## Acceptance Criteria

```gherkin
Feature: IR metrics
  Scenario: Run BEIR on small dataset
    Given I run tools/eval/run_beir.py
    Then a CSV SHALL contain recall@k and nDCG@10
```

## References

- BEIR repo; RAGAS docs.

## CLI Outputs & Offline Requirements

- IR (BEIR) CLI MUST write a `leaderboard.csv` containing columns including (but not limited to): `NDCG@10`, `Recall@10`, `MRR@10`.
- E2E (RAGAS) CLI MUST write a `leaderboard.csv` containing columns including (but not limited to): `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
- Tests MUST be deterministic and offline:
  - Heavy network/dataset downloads are NOT allowed in CI; use strict mocks or tiny local datasets.
  - Coordinator/retriever interactions MUST be stubbed.
