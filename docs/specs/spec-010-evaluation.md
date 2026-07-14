---
spec: SPEC-010
title: Evaluation: BEIR/M-BEIR for Offline Retrieval Quality
version: 1.1.0
date: 2026-07-13
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-EVAL-001: Provide offline scripts to compute recall@k, nDCG, MRR on BEIR/M-BEIR.
  - NFR-PERF-004: Track latency and memory budgets per profile.
related_adrs: ["ADR-011"]
---


## Objective

Ship one offline evaluation path for retrieval quality. Record deterministic
BEIR metrics in a small leaderboard CSV for regressions.

## Libraries and Imports

```python
from beir.datasets.data_loader import GenericDataLoader  # or datasets from HF
```

## File Operations

### CREATE

- `tools/eval/run_beir.py`.
- `data/eval/README.md` with dataset instructions.

## Acceptance Criteria

```gherkin
Feature: IR metrics
  Scenario: Run BEIR on small dataset
    Given I run tools/eval/run_beir.py
    Then a CSV SHALL contain recall@k and nDCG@10
```

## References

- BEIR repository and documentation.

## CLI Outputs & Offline Requirements

- IR (BEIR) CLI MUST write a `leaderboard.csv` with fields: `schema_version`, `ts`, `dataset`, `k`, and dynamic metric columns `ndcg@{k}`, `recall@{k}`, `mrr@{k}` plus `sample_count`.
- Determinism: CLIs MUST set seeds and thread caps; support `--sample_count` for deterministic subsets.
- Tests MUST be deterministic and offline:
  - Heavy network/dataset downloads are NOT allowed in CI; use strict mocks or tiny local datasets.
  - Retriever interactions MUST be stubbed.

RAGAS is not a supported DocMind dependency or evaluation path. The previous
CLI was removed because it could not run from the published `eval` extra and
its tests skipped when RAGAS was absent.
