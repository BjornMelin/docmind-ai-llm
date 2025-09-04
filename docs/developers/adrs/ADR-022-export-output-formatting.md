---
ADR: 022
Title: Export & Structured Output System
Status: Accepted
Version: 2.1
Date: 2025-08-19
Supersedes:
Superseded-by:
Related: 013, 024
Tags: export, structured-output
References:
- [Pydantic v2 — BaseModel](https://docs.pydantic.dev/latest/)
---

## Description

Provide structured outputs (JSON) and simple export paths (Markdown/PDF) using Pydantic v2 models and small helpers.

## Context

Users need reliable, typed output for downstream use and lightweight export.

## Decision Drivers

- Typed models; predictable schemas
- Minimal code and dependencies

## Alternatives

- Ad‑hoc dicts — brittle
- Heavy export stack — unnecessary

### Decision Framework

| Option         | Robustness (40%) | Simplicity (40%) | Perf (20%) | Total | Decision |
| -------------- | ---------------- | ---------------- | ---------- | ----- | -------- |
| Pydantic v2    | 9                | 9                | 9          | 9.0   | ✅ Sel.  |

## Decision

Use Pydantic models at boundaries; export small Markdown/PDF via standard libs.

## High-Level Architecture

LLM → Pydantic validation → export helper

## Related Requirements

### Functional Requirements

- FR‑1: Typed structured outputs for downstream use
- FR‑2: Export to Markdown/PDF

### Non-Functional Requirements

- NFR‑1: Keep helpers minimal; no heavy export stack

### Performance Requirements

- PR‑1: Export completes under 200ms for typical answer sizes

### Integration Requirements

- IR‑1: Pydantic v2 models at boundaries
- IR‑2: Single helper for exports

## Design

### Implementation Details

```python
from pydantic import BaseModel

class Answer(BaseModel):
    text: str
    sources: list[str]

def export_markdown(ans: Answer, path: str) -> str:
    md = f"# Answer\n\n{ans.text}\n\n## Sources\n" + "\n".join(f"- {s}" for s in ans.sources)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path
```

### Configuration

```env
DOCMIND_EXPORT__DEFAULT_FORMAT=markdown
```

## Testing

- Validate model parsing; round‑trip export/import

```python
def test_answer_model_roundtrip():
    a = Answer(text="t", sources=["s"]) 
    j = a.model_dump(mode="json")
    assert j["text"] == "t"
```

## Consequences

### Positive Outcomes

- Stable output for integrations

### Negative Consequences / Trade-offs

- Requires schema evolution discipline

### Dependencies

- Python: `pydantic>=2`

## Changelog

- 2.1 (2025‑09‑04): Standardized to template; added requirements/config/tests
- 2.0 (2025‑08‑19): Accepted; structured outputs defined via Pydantic
