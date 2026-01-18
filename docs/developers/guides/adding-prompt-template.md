# Developer Guide — Adding a Prompt Template (SPEC‑020)

This guide explains how to add a new prompt template to DocMind AI’s file‑based prompt system (SPEC‑020). The system uses LlamaIndex’s `RichPromptTemplate` (Jinja syntax under the hood) with YAML front matter for metadata.

## Where templates live

- Templates: `templates/prompts/*.prompt.md`
- Presets (optional): `templates/presets/{tones,roles,lengths}.yaml`

## Template format

A template file contains a YAML front matter header followed by a Jinja body. Front matter keys:

- `id` (str): Unique identifier (used by code)
- `name` (str): Human‑readable name (shown in UI)
- `description` (str): Short description
- `tags` (list[str], optional)
- `required` (list[str], optional): Variables that must be provided at render time
- `defaults` (dict, optional): Default variable values
- `version` (int): Simple version for telemetry and evolution

Example:

```yaml
---
id: summary-open-questions
name: Summarize and Identify Open Questions
description: Summarize content and list unresolved questions
required: [context]
defaults:
  tone: { description: "Use a neutral, objective tone." }
version: 1
---
{% chat role="system" %}
{{ tone.description }}
{% endchat %}

{% chat role="user" %}
Summarize the content and list any open questions.

<context>
{{ context }}
</context>
{% endchat %}
```

## Presets (optional)

Presets provide named options for tone, role, and length. They’re simple YAML dictionaries under `templates/presets/`:

```yaml
# templates/presets/tones.yaml
professional:
  description: Use a professional, objective tone.
```

You can reference presets in your template context from the UI or programmatically.

## Programmatic usage

```python
from src.prompting import list_templates, render_prompt, format_messages

# Select a template
tpl = next(t for t in list_templates() if t.id == "summary-open-questions")

# Build context
ctx = {
  "context": "…",
  "tone": {"description": "Use a neutral tone."}
}

# Render
text_prompt = render_prompt(tpl.id, ctx)
chat_msgs = format_messages(tpl.id, ctx)
```

## Validation & lint

- The loader performs basic validation on front matter.
- Jinja meta checks are available to detect undeclared variables, but render‑time checks are the primary guard.
- Keep templates concise and clear; avoid unnecessary logic. Do not access the filesystem from templates.

## Testing

- Add or update unit tests under `tests/unit/prompting/` if you add new template conventions.
- The default smoke tests validate that at least one template is present and renderable with defaults.

## UI visibility

- The entrypoint `app.py` launches the Streamlit app in `src/app.py`; the app
  lists templates by `name` and uses `id` to render prompts.
- If the template is not visible in the UI, ensure it has valid front matter and a body, and restart the app.
