---
ADR: 014
Title: Testing and Quality Validation (pytest + DeepEval)
Status: Accepted
Version: 1.1
Date: 2025-09-03
Supersedes:
Superseded-by:
Related: 012, 003, 004, 010
Tags: testing, quality, pytest, deepeval
References:
- [pytest — Official Docs](https://docs.pytest.org/)
- [DeepEval — Documentation](https://docs.confident-ai.com/)
---

## Description

Define a simple, local‑first testing stack: pytest for orchestration, DeepEval for RAG metrics, and small fixtures/datasets. Keep tests fast and deterministic.

## Context

DocMind AI combines agentic flows, retrieval, and local LLMs. We need practical validation without heavy harnesses or external services.

## Decision Drivers

- Deterministic local runs
- Standard tools; minimal custom code
- Clear pass/fail gates

## Alternatives

- Manual QA only — non‑scalable
- Custom test harness — high effort
- DeepEval + pytest (Selected) — balanced capability/simplicity

### Decision Framework

| Option            | Simplicity (40%) | Capability (30%) | CI Fit (20%) | Effort (10%) | Total | Decision    |
| ----------------- | ---------------- | ---------------- | ------------ | ------------ | ----- | ----------- |
| DeepEval + pytest | 9                | 9                | 9            | 9            | 9.0   | ✅ Selected |
| Custom harness    | 4                | 8                | 6            | 3            | 5.5   | Rejected    |

## Decision

Use pytest as the single runner; integrate DeepEval for RAG‑specific metrics; keep small datasets under `tests/data`.

## High-Level Architecture

pytest → fixtures → evaluation helpers → metrics/thresholds

## Related Requirements

### Functional Requirements

- FR‑1: Retrieval quality thresholds
- FR‑2: Answer faithfulness thresholds
- FR‑3: Performance budgets (latency/tokens)

### Non-Functional Requirements

- NFR‑1: Offline, local; reproducible
- NFR‑2: Unit <5s; integration <30s; system <5m

### Performance Requirements

- PR‑1: RAG quality suite completes under 5 minutes locally
- PR‑2: Latency probe asserts P95 budget per ADR‑010

### Integration Requirements

- IR‑1: pytest markers `unit|integration|system` honored
- IR‑2: CI emits junitxml and JSON report artifacts
- IR‑3: Tests follow boundary‑first strategy per ADR‑029; internal mocks minimized

## Design

### Architecture Overview

- tests → fixtures → DeepEval helpers → metrics → reports

### Implementation Details

```python
# tests/test_quality.py (skeleton)
import pytest

@pytest.mark.unit
def test_retrieval_thresholds(rag_eval):
    res = rag_eval.evaluate([{"query":"Q","answer":"A","contexts":["C"]}])
    assert res.metrics["precision"] >= 0.6
```

### Extended Implementation

```python
# Golden dataset pattern + DeepEval
from pathlib import Path
import json
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

def load_golden(path: str = "tests/data/golden.json") -> list[LLMTestCase]:
    data = json.loads(Path(path).read_text())
    cases = []
    for c in data.get("cases", []):
        cases.append(LLMTestCase(
            input=c["query"],
            expected_output=c.get("expected"),
            actual_output=c.get("actual", ""),
            retrieval_context=c.get("contexts", []),
        ))
    return cases

def test_golden_quality():
    metrics = [AnswerRelevancyMetric(threshold=0.7), FaithfulnessMetric(threshold=0.8)]
    for case in load_golden():
        assert_test(case, metrics)
```

### CI Outline

- Run unit/integration first; then RAG quality with CPU‑only flags
- Upload JSON reports for inspection

### Configuration

- Markers: unit, integration, system
- Thresholds set per ADR‑012

## Testing

- Parametrize canonical queries; verify thresholds
- Run in CI with `uv run pytest -q` (note: `-q` for clean output; use `-v` for verbose debugging)

## Consequences

### Positive Outcomes

- Clear quality bars; fast feedback
- Minimal maintenance; standard tools

### Negative Consequences / Trade-offs

- Threshold tuning required as models evolve

### Ongoing Maintenance & Considerations

- Review thresholds quarterly to avoid flakiness
- Keep golden datasets small; pin seeds and hashes where applicable

### Dependencies

- Python: `pytest>=8`, `deepeval` (pinned)

## Changelog

- 1.1 (2025‑09‑03): Accepted; aligned with ADR‑012 gates
