---
ADR: 018
Title: Automatic Prompt Optimization with DSPy
Status: Implemented
Version: 2.1
Date: 2025-08-19
Supersedes:
Superseded-by:
Related: 001, 003, 020
Tags: prompts, dspy, optimization
References:
- DSPy documentation
---

## Description

Use DSPy to automatically optimize prompts and rewrite queries. Replace ad‑hoc prompt tuning with programmatic, data‑driven compilation.

## Context

Manual prompt engineering is brittle and slow. DSPy compiles prompt programs using examples and metrics, improving retrieval and answer quality.

## Decision Drivers

- Library‑first; remove custom tuning code
- Measurable improvements; reproducibility

## Alternatives

- Manual prompts — slow, inconsistent
- Custom optimizer — high effort

### Decision Framework

| Option | Quality (50%) | Effort (30%) | Maintainability (20%) | Total | Decision |
| ------ | ------------- | ------------ | --------------------- | ----- | -------- |
| DSPy   | 9             | 8            | 9                     | 8.8   | ✅ Sel.  |

## Decision

Adopt DSPy for prompt/query optimization; integrate with retrieval where beneficial.

## High-Level Architecture

Queries/examples → DSPy compile → Optimized prompts → Retrieval/answer

## Related Requirements

### Functional Requirements

- FR‑1: Optimize prompts/queries automatically based on metrics
- FR‑2: Integrate with retrieval to improve recall/precision

### Non-Functional Requirements

- NFR‑1: Reproducible runs; deterministic defaults
- NFR‑2: Minimal code additions

### Performance Requirements

- PR‑1: Compile step completes under 2 minutes on a sample set

### Integration Requirements

- IR‑1: Controlled via settings flag
- IR‑2: Exposes an `optimize_query()` helper

## Design

### Implementation Details

```python
# skeleton
def optimize_query(query: str) -> str:
    # call into dspy optimizer; return improved query
    return query
```

### Configuration

```env
DOCMIND_DSPY__ENABLED=false
```

## Testing

- A/B evaluate retrieval before/after optimization (ADR‑012 thresholds)

```python
def test_optimize_query_noop_when_disabled(settings):
    assert optimize_query("q") == "q"
```

## Consequences

### Positive Outcomes

- Better retrieval/answers with minimal code

### Negative Consequences / Trade-offs

- Requires curated examples to reach best gains

### Dependencies

- Python: `dspy` (pinned)

## Changelog

- 2.1 (2025‑09‑04): Standardized to template; added requirements/config/tests
- 2.0 (2025‑08‑19): Implemented DSPy integration
