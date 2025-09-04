---
ADR: 012
Title: Evaluation Strategy with DeepEval + Pytest
Status: Accepted
Version: 4.1
Date: 2025-08-19
Supersedes:
Superseded-by:
Related: 001, 003, 004, 010, 014
Tags: evaluation, quality, pytest, deepeval
References:
- [DeepEval — Documentation](https://docs.confident-ai.com/)
---

## Description

Use DeepEval for RAG‑specific evaluation integrated with pytest. Aligns with local‑first constraints and current 128K context (ADR‑004/010). Covers retrieval quality, answer faithfulness, and latency regressions.

## Context

Prior ADR contained extensive custom evaluation code not needed. DeepEval provides production metrics (faithfulness, context precision/recall, coherence) with simple usage patterns and good CI fit via pytest.

## Decision Drivers

- Library‑first; avoid custom metric code
- Deterministic, offline execution
- Clear thresholds; actionable reports

## Alternatives

- Manual evaluation — subjective, not scalable
- RAGAS — capable, less seamless pytest fit
- Custom pytest metrics — high effort, reinventing wheels

### Decision Framework

| Option                 | Capability (40%) | Simplicity (30%) | CI Fit (20%) | Effort (10%) | Total | Decision      |
| ---------------------- | ---------------- | ---------------- | ------------ | ------------ | ----- | ------------- |
| DeepEval + pytest      | 9                | 9                | 9            | 9            | 9.0   | ✅ Selected    |
| RAGAS                  | 8                | 7                | 7            | 7            | 7.6   | Rejected      |
| Custom metrics         | 7                | 4                | 8            | 3            | 5.9   | Rejected      |

## Decision

Use DeepEval for metrics and pytest for orchestration. Keep tests small and deterministic; run fully offline.

## High-Level Architecture

Tests → DeepEval metrics → Reports (CI) → Fix/iterate

## Related Requirements

### Functional Requirements

- FR‑1: Evaluate retrieval (recall/precision, coverage)
- FR‑2: Evaluate answers (faithfulness, coherence)
- FR‑3: Track latency and token usage

### Non-Functional Requirements

- NFR‑1: Local‑first; no external API
- NFR‑2: Tests run in <10m on dev hardware

### Performance Requirements

- PR‑1: Eval suites complete under 5m for unit/integration tier
- PR‑2: Latency regression checks compare P50/P95 vs baseline

### Integration Requirements

- IR‑1: pytest markers `unit`, `integration`, `system` respected
- IR‑2: Outputs junitxml and simple JSON report for CI

## Design

### Implementation Details

```python
# tests/eval/test_rag_eval.py (skeleton)
import pytest

def test_retrieval_quality(rag_evaluator):
    result = rag_evaluator.evaluate([{"query": "Q", "answer": "A", "contexts": ["C"]}])
    assert result.metrics["recall"] >= 0.7
```

```python
# tests/eval/test_latency.py (skeleton)
def test_latency_budget(latency_probe):
    p95 = latency_probe.measure_p95()
    assert p95 < 1.5  # seconds
```

### Configuration

- Place evaluation fixtures under `tests/conftest.py`
- Keep sample datasets under `tests/data/`

```ini
# pytest.ini
[pytest]
markers = unit integration system
addopts = -q --disable-warnings --maxfail=1
```

```env
DOCMIND_EVAL__THRESHOLD_RECALL=0.7
DOCMIND_EVAL__THRESHOLD_LATENCY_P95=1.5
```

## Testing

- Parametrize representative queries; assert metric thresholds
- Smoke test latency budget on local hardware

## Consequences

### Positive Outcomes

- Standard metrics; minimal code
- Good CI integration via pytest

### Negative Consequences / Trade-offs

- Metric thresholds require tuning to avoid flaky tests

### Dependencies

- Python: `deepeval` (pinned), `pytest>=8`

## Changelog

- 4.1 (2025‑09‑04): Standardized to template; added PR/IR, pytest config/tests; no behavior change
- 4.0 (2025‑08‑19): Accepted; updated to align with 128K context (replaces 262K mentions)
