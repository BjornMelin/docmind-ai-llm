---
ADR: 023
Title: Document Analysis Mode Strategy
Status: Accepted
Version: 2.1
Date: 2025-08-19
Supersedes:
Superseded-by:
Related: 001, 003, 013
Tags: analysis, modes
References:
- [Project ADRs README](./README.md)
---

## Description

Define a small set of analysis modes (e.g., summary, deep‑dive, compare) mapped to retrieval/LLM settings. Expose via UI.

## Context

Different tasks need different retrieval depth and response style.

## Decision Drivers

- Clear UX; minimal settings surface
- Reuse existing pipeline knobs

## Alternatives

- Many hidden toggles — confusing
- Single mode — inflexible

### Decision Framework

| Option           | UX (40%) | Simplicity (40%) | Coverage (20%) | Total | Decision |
| ---------------- | -------- | ---------------- | -------------- | ----- | -------- |
| Few named modes  | 9        | 9                | 8              | 8.8   | ✅ Sel.  |

## Decision

Implement 3–4 named modes with sensible defaults for retrieval depth, reranking, and output format.

## High-Level Architecture

Mode → preset → pipeline settings → response

## Related Requirements

### Functional Requirements

- FR‑1: Provide 3–4 named analysis modes
- FR‑2: Each mode maps to retrieval/reranking presets

### Non-Functional Requirements

- NFR‑1: Clear UX; easy to extend

### Performance Requirements

- PR‑1: Mode switching has no measurable UI latency (>100ms)

### Integration Requirements

- IR‑1: Exposed in UI (ADR‑013)
- IR‑2: Backed by settings registry (ADR‑024)

## Design

### Architecture Overview

- Mode → preset → pipeline knobs (retrieval depth, reranker, output style)

### Implementation Details

```python
ANALYSIS_MODES = {
  "summary": {"top_k": 5},
  "deep": {"top_k": 20},
}
```

### Configuration

```env
DOCMIND_ANALYSIS__DEFAULT_MODE=summary
```

## Testing

- Snapshot outputs per mode for a small corpus

```python
@pytest.mark.unit
def test_default_mode(settings):
    assert settings.analysis.default_mode == "summary"
```

## Consequences

### Positive Outcomes

- Predictable behavior; easy to document

### Negative Consequences / Trade-offs

- More presets to maintain

### Dependencies

- None specific

### Ongoing Maintenance & Considerations

- Keep number of modes low; ensure each has clear, documented behavior

## Changelog

- 2.1 (2025‑09‑04): Standardized to template; added requirements/config/tests

- 2.0 (2025‑08‑19): Accepted; minimal named modes
