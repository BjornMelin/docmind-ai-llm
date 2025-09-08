---
ADR: 020
Title: Prompt Template System (RichPromptTemplate, File-Based)
Status: Accepted
Version: 3.0
Date: 2025-09-08
Supersedes:
Superseded-by:
Related: 004, 018, 020 (SPEC)
Tags: prompts, templates, jinja, llamaindex
References:
- [LlamaIndex — Prompt Templates](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/)
---

## Description

Provide a small, file‑based prompt template system using LlamaIndex `RichPromptTemplate` (Jinja under the hood) with YAML front matter metadata and presets for tones/roles/lengths. Aligns with 128K context (ADR‑004/010).

## Context

Hard‑coded dictionaries (`src/prompts.py`) are brittle and not extensible. We want configurable, auditable templates stored on disk, rendered through a library‑first API, without maintaining our own Jinja environment.

## Decision Drivers

- Flexibility with minimal surface area
- Library‑first (no custom Jinja env), fast iteration
- KISS/DRY/YAGNI — only thin glue where necessary

## Alternatives

- Hard‑coded prompts — brittle
- Custom Jinja environment — unnecessary complexity

### Decision Framework

| Option                 | Flex (40%) | Simplicity (40%) | Perf (20%) | Total | Decision |
| ---------------------- | ---------- | ---------------- | ---------- | ----- | -------- |
| RichPromptTemplate + FS| 9          | 9                | 9          | 9.0   | ✅ Sel.  |

## Decision

Use file‑based templates rendered via LlamaIndex `RichPromptTemplate`. Keep a thin loader/registry, and store tones/roles/lengths as YAML presets.

## High-Level Architecture

Templates on disk → registry → RichPromptTemplate.format/format_messages → call LLM

## Related Requirements

### Functional Requirements

- FR‑1: Support multiple prompt types with metadata
- FR‑2: Support tones/roles/lengths via presets

### Non-Functional Requirements

- NFR‑1: Minimal templating layer; easy to audit
- NFR‑2: Library‑first (`RichPromptTemplate`), zero custom Jinja env

### Performance Requirements

- PR‑1: Template render <5ms typical; compile once per id and reuse

### Integration Requirements

- IR‑1: Centralized template registry (lazy load)
- IR‑2: ADR‑024 settings drive defaults

## Design

### Architecture Overview

- Template/preset files on disk
- Loader parses YAML front matter + body
- Registry caches `TemplateSpec` + compiled `RichPromptTemplate`
- Renderer provides `render_prompt()` and `format_messages()`

### Implementation Details

```python
from llama_index.core.prompts import RichPromptTemplate

BODY = """
{% chat role="system" %}You are {{ role.description }}. Use a {{ tone.description }} tone.{% endchat %}
{% chat role="user" %}Context:\n{{ context }}\nTask: {{ task }}{% endchat %}
"""

TPL = RichPromptTemplate(BODY)

def format_messages(context: str, task: str, role: dict, tone: dict):
    return TPL.format_messages(context=context, task=task, role=role, tone=tone)
```

```python
# Registry API (conceptual)
from src.prompting import list_templates, get_template, render_prompt, format_messages, list_presets

templates = list_templates()               # -> [TemplateMeta]
spec = get_template("comprehensive-analysis")
messages = format_messages(
  "comprehensive-analysis",
  {"context": ctx, "task": "analyze", "role": presets["roles"]["researcher"], "tone": presets["tones"]["professional"]},
)
```

### Configuration

```env
DOCMIND_PROMPT__DEFAULT_TONE=professional
DOCMIND_PROMPT__DEFAULT_ROLE=researcher
```

## Testing

- Unit + integration tests validate loader, registry, and render/format_messages; E2E validates UI selection and rendering.

## Consequences

### Positive Outcomes

- Clear, configurable prompts; easy to extend; minimal glue code

### Negative Consequences / Trade-offs

- Requires template catalog discipline + small lint/validation layer

### Dependencies

- Python: `llama-index-core` (RichPromptTemplate), `jinja2` (as dependency)

### Ongoing Maintenance & Considerations

- Keep template set small; document options; version templates when schema changes

## Changelog

- 3.0 (2025‑09‑08): Library‑first redesign using LlamaIndex `RichPromptTemplate`; file‑based templates and presets; minimal registry; render + format_messages APIs
- 2.1 (2025‑09‑04): Standardized to template; added requirements/config/tests
- 2.0 (2025‑08‑19): Accepted; aligned to 128K window
