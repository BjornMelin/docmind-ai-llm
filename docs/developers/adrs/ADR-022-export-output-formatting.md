---
ADR: 022
Title: Export & Structured Output System
Status: Accepted
Version: 1.0
Date: 2025-08-18
Supersedes:
Superseded-by:
Related: 001, 016, 020, 021, 024
Tags: export, json, markdown, templates
References:
- [Pydantic](https://docs.pydantic.dev/)
- [Jinja2](https://jinja.palletsprojects.com/)
- [Rich](https://rich.readthedocs.io/)
---

## Description

Provide type‑safe export of analysis results to JSON and Markdown with optional Rich console output. Use existing Pydantic models and simple templates.

## Context

Users need portable results (JSON) and readable docs (Markdown). We can leverage existing models to avoid bespoke formatters.

## Decision Drivers

- Type‑safe serialization and minimal glue
- Customizable formatting where needed
- Fast exports with small footprint

## Alternatives

- A: Manual string formatting — Error‑prone
- B: Heavy report libs — Overkill for local app
- C: Pydantic + Jinja2 + Rich (Selected) — Lightweight and extensible

### Decision Framework

| Model / Option              | Type-Safety (35%) | Simplicity (35%) | Flexibility (20%) | Performance (10%) | Total | Decision      |
| --------------------------- | ----------------- | ---------------- | ----------------- | ----------------- | ----- | ------------- |
| Pydantic+Jinja2+Rich (Sel.) | 10                | 9                | 8                 | 8                 | **9.0** | ✅ Selected    |
| Manual strings              | 3                 | 9                | 5                 | 9                 | 5.7   | Rejected      |
| Heavy libs                  | 8                 | 5                | 9                 | 6                 | 6.9   | Rejected      |

## Decision

Build a small Export Manager that renders JSON (model_dump) and Markdown (Jinja2). Keep template registry simple.

## High-Level Architecture

```mermaid
graph TD
  R[Results (Pydantic)] --> E[Export Manager]
  E --> J[JSON]
  E --> M[Markdown]
  E --> C[Rich Console]
```

## Related Requirements

### Functional Requirements

- FR‑1: JSON export with metadata
- FR‑2: Markdown export via templates
- FR‑3: Batch export support

### Non-Functional Requirements

- NFR‑1: <500ms typical export time
- NFR‑2: Preserve types/structure

### Integration Requirements

- IR‑1: UI triggers (ADR‑013/016)

## Design

### Architecture Overview

- Export Manager serializes Pydantic models to JSON and renders Markdown via templates; optional Rich console output.

### Implementation Details

In `src/export/manager.py` (illustrative):

```python
from typing import Any
from jinja2 import Template

class ExportManager:
    def to_json(self, model: Any) -> dict:
        return model.model_dump(mode="json")

    def to_markdown(self, model: Any, tmpl_str: str) -> str:
        return Template(tmpl_str).render(obj=model)

    def to_markdown_standard(self, model: Any) -> str:
        tmpl = """
        # {{ obj.title }}\n\n
        **Summary**: {{ obj.summary }}\n\n
        {% for item in obj.items %}- {{ item }}\n{% endfor %}
        """
        return self.to_markdown(model, tmpl)
```

Additional helpers:

```python
from typing import Any
from rich.console import Console
from rich.table import Table

class ExportManager:
    # ... previous methods ...
    def to_rich_console(self, model: Any) -> str:
        table = Table(title=getattr(model, "title", "Result"))
        table.add_column("Field")
        table.add_column("Value")
        for k, v in model.model_dump(mode="json").items():
            table.add_row(str(k), str(v))
        console = Console(record=True)
        console.print(table)
        return console.export_text()

def batch_export_json(models: list[Any]) -> list[dict]:
    mgr = ExportManager()
    return [mgr.to_json(m) for m in models]
```

### Configuration

No new settings.

## Testing

```python
def test_json_roundtrip(example_model):
    assert to_json(example_model)
```

## Consequences

### Positive Outcomes

- Simple, type‑safe exports
- Customizable Markdown via templates

### Negative Consequences / Trade-offs

- Template maintenance when schema changes

### Ongoing Maintenance & Considerations

- Keep templates minimal; update alongside models

### Dependencies

- Python: `pydantic`, `jinja2`, `rich`

## Changelog

- 1.0 (2025-08-18): Initial export system and formats
