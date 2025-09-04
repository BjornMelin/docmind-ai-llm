---
ADR: 020
Title: Dynamic Prompt Template & Response Configuration
Status: Accepted
Version: 2.1
Date: 2025-08-19
Supersedes:
Superseded-by:
Related: 004, 018
Tags: prompts, templates, jinja
References:
- [LlamaIndex — Prompt Templates](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/)
---

## Description

Provide a small, dynamic template system (types, tones, roles, detail levels). Aligns with 128K context (ADR‑004/010), not 262K.

## Context

Users need configurable prompts without bespoke code per combination.

## Decision Drivers

- Flexibility with minimal surface area
- Clear configuration; fast iteration

## Alternatives

- Hard‑coded prompts — brittle
- Overly generic builders — complex

### Decision Framework

| Option        | Flex (40%) | Simplicity (40%) | Perf (20%) | Total | Decision |
| ------------- | ---------- | ---------------- | ---------- | ----- | -------- |
| Small Jinja   | 9          | 9                | 9          | 9.0   | ✅ Sel.  |

## Decision

Use a small Jinja template layer with a single settings object mapping desired options.

## High-Level Architecture

Settings → select template → render → call LLM

## Related Requirements

### Functional Requirements

- FR‑1: Support multiple prompt types/tones/roles
- FR‑2: Map mode to LLM params cleanly

### Non-Functional Requirements

- NFR‑1: Minimal templating layer; easy to audit

### Performance Requirements

- PR‑1: Template render <5ms typical

### Integration Requirements

- IR‑1: Centralized template registry
- IR‑2: ADR‑024 settings drive defaults

## Design

### Implementation Details

```python
from jinja2 import Template

PROMPT = Template("""
You are {{ role }}. Answer with a {{ tone }} tone.
Question: {{ question }}
""")

def render_prompt(question, tone="neutral", role="analyst"):
    return PROMPT.render(question=question, tone=tone, role=role)
```

### Template Registry and LLM Param Mapping

```python
# simple registry
TEMPLATES = {
    "qa": Template("You are {{ role }}. Answer {{ style }}.\nQ: {{ q }}"),
    "summarize": Template("Summarize in a {{ detail }} style.\nText: {{ q }}"),
}

LLM_PROFILES = {
    # map modes to generation params
    "concise": {"temperature": 0.2, "max_tokens": 512},
    "detailed": {"temperature": 0.4, "max_tokens": 2048},
}

def build_prompt(mode: str, q: str, role="analyst", style="concisely", detail="concise"):
    tpl = TEMPLATES[mode]
    text = tpl.render(q=q, role=role, style=style, detail=detail)
    params = LLM_PROFILES[detail]
    return text, params
```

### Structured Output Pattern

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    sources: list[str] = []

STRUCTURED_TEMPLATE = Template(
    """Return JSON with keys: answer, sources.\nQuestion: {{ q }}"""
)

def render_structured(q: str):
    return STRUCTURED_TEMPLATE.render(q=q)
```

### Configuration

```env
DOCMIND_PROMPT__DEFAULT_TONE=neutral
DOCMIND_PROMPT__DEFAULT_ROLE=analyst
```

## Testing

- Unit test renders for a few combinations

```python
def test_render_prompt_defaults():
    p = render_prompt("Q")
    assert "Question: Q" in p
```

## Consequences

### Positive Outcomes

- Clear, configurable prompts; easy to extend

### Negative Consequences / Trade-offs

- Requires basic docs for options

### Dependencies

- Python: `jinja2`

## Changelog

- 2.1 (2025‑09‑04): Standardized to template; added requirements/config/tests
- 2.0 (2025‑08‑19): Accepted; aligned to 128K window
