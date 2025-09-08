# SPEC-020 — Prompt Template System (Full Replacement)

Status: Planned (Breaking)

Owners: Retrieval/UX Core

Related: ADR-020 (Prompt Template System), ADR-003 (Adaptive Retrieval), ADR-004 (Local-First LLM), ADR-024 (Always-On Hybrid + Rerank)

## Summary

Replace the current hard-coded prompt constants (`src/prompts.py`) with a file-based prompt template system built on LlamaIndex’s `RichPromptTemplate` (Jinja under the hood). The new system provides:

- First-class, versioned templates with metadata (YAML front matter)
- Strong typing and validation for required variables
- Library-first rendering via LlamaIndex `RichPromptTemplate` with Jinja semantics
- Clear extension points (custom templates, tags, presets) without touching code

This is a breaking change that fully deletes legacy prompt constants and their tests, and updates the UI and agents to load templates from the new registry. No backwards compatibility shims or dual paths will be kept.

## Goals

- Configurable, reusable prompt templates decoupled from code
- Strong validation: fail fast on missing/unknown variables
- Local-first: no network at import or render time
- Predictable performance with caching
- Simple developer workflow to add/modify templates

## Non‑Goals

- No runtime template editing UI in this phase
- No remote repository sync (Git-backed catalogs can be added later)
- No i18n layer in v1 (we will keep templates English-only initially)

## Design Overview

### Directory Layout (Small, Library-First)

```
src/
  prompting/
    __init__.py
    models.py          # Pydantic models: TemplateMeta, TemplateSpec
    loader.py          # Disk loader with cache + validation
    registry.py        # In-memory registry + query APIs
    renderer.py        # Thin adapter around LlamaIndex RichPromptTemplate
    validators.py      # Static/semantic checks, lint helpers
templates/
  prompts/
    comprehensive-analysis.prompt.md
    key-insights.prompt.md
    summary-open-questions.prompt.md
  presets/
    tones.yaml         # Named style hints (replaces TONES)
    roles.yaml         # Role instructions (replaces INSTRUCTIONS)
    lengths.yaml       # Output length preferences (replaces LENGTHS)
```

### Template Format (Markdown + YAML front matter)

```
---
id: comprehensive-analysis
name: Comprehensive Document Analysis
description: Summary, key insights, action items, open questions
tags: [analysis, default]
required:
  - context
  - style
defaults:
  style: professional
version: 1
---
You are an expert assistant.

Context:
{{ context }}

Style: {{ style.description }}

Tasks:
- Provide a summary
- Extract key insights
- List action items
- Raise open questions

Answer clearly and concisely.
```

- Front matter is required. Keys:
  - `id`, `name`, `description`, `tags[]`, `required[]`, `defaults{}`, `version`
- Body uses LlamaIndex `RichPromptTemplate` Jinja syntax. Variables must be declared in `required` or `defaults`.

### Presets (YAML)

Example `templates/presets/tones.yaml`:

```
professional:
  description: Use a professional, objective tone.
academic:
  description: Use a scholarly, research‑oriented tone.
informal:
  description: Use a casual, conversational tone.
```

Similar files for roles (replaces INSTRUCTIONS) and lengths.

### Public API (Minimal Surface)

```
from src.prompting import (
  list_templates,        # -> list[TemplateMeta]
  get_template,          # (id) -> TemplateSpec
  render_prompt,         # (id, context: dict) -> str
  format_messages,       # (id, context: dict) -> list[BaseMessage]
  list_presets,          # (kind: Literal["tones","roles","lengths"]) -> dict
)
```

### Rendering Pipeline (Leverage LlamaIndex)

1. Loader scans `templates/prompts/*.prompt.md` on first use
2. Parse YAML front matter + Jinja body; build `TemplateSpec`
3. Validate (keep simple, no custom engine):
   - required variables present in context (fast check)
   - optional static check: Jinja meta `find_undeclared_variables()` to detect missing vars early
4. Wrap body in `RichPromptTemplate` for rendering
5. Registry caches `TemplateSpec` + `RichPromptTemplate` instances by `id`
6. `render_prompt(id, context)` merges defaults + context and calls `RichPromptTemplate.format(**ctx)`
7. `format_messages(id, context)` calls `RichPromptTemplate.format_messages(**ctx)` for chat LLMs

### Jinja & Safety

- Use LlamaIndex `RichPromptTemplate` (ships with Jinja) instead of managing our own environment
- Optional static validation via `jinja2.Environment().parse` + `meta.find_undeclared_variables` for missing vars
- No custom filters initially (YAGNI). Add only if a concrete template requires it.

### Validation & Lint (KISS)

- Front matter required keys present: `id`, `name`, `description`, `version`
- Required variables satisfied at render time; static check optional but supported
- Body length sanity check (> 40 chars) to avoid empty templates
- Unit tests assert catalog validity (offline, no network)

## UI Integration (Streamlit)

- Replace `from src.prompts import PREDEFINED_PROMPTS` with registry APIs
- Dropdowns:
  - Template selector from `list_templates()` (id/name)
  - Optional style/role/length selectors backed by YAML presets
- Render path:
  - Build context dict: `context` (document summary or selection), `style` (object from presets), `role` (optional), `length` (optional)
  - Prefer `format_messages(template_id, context)` for chat LLMs; fall back to `render_prompt` for completion LLMs
- Telemetry:
  - Log `prompt.template_id`, `prompt.version`

## Deletions (No Backcompat)

- Remove `src/prompts.py`
- Remove tests reading from `src/prompts`:
  - `tests/unit/prompts/test_prompts.py`
  - E2E checks that refer to PREDEFINED_PROMPTS (update to new registry checks)
- Update `src/app.py` to new API

## New Tests (Library-First)

- `tests/unit/prompting/test_loader.py` — parses and validates default templates
- `tests/unit/prompting/test_renderer.py` — renders with defaults; missing vars raises (KeyError or Jinja Undefined)
- `tests/integration/test_prompt_registry.py` — list/select/render happy path
- `tests/e2e/test_prompt_system.py` — end‑to‑end UI interactions (template count, selection, render)

## Security Considerations

- Use `RichPromptTemplate`; do not expose arbitrary Python in templates
- Optional StrictUndefined behavior enforced via pre-validate (missing vars cause errors)
- No file system access in templates; templates are data only

## Performance Considerations

- Catalog loads on first call and caches in memory
- `RichPromptTemplate` is compiled once per template id and reused

## Rollout & Breaking Changes

- Single big‑bang PR (no toggles):
  - Add prompting subsystem + default templates + UI integration
  - Remove legacy module and tests
  - Update docs (README “Choosing Prompts”), ADR‑020 references

## Implementation Plan (Work Breakdown — Minimal Code)

1) Scaffolding (src/prompting)
- Add `models.py` (TemplateMeta, TemplateSpec) — Pydantic for YAML front matter
- Add `loader.py` (scan/parse, front matter, minimal validation)
- Add `renderer.py` (wrap `RichPromptTemplate`; expose render + format_messages)
- Add `registry.py` (in‑memory dict; list/get APIs; lazy load on first use)
- Add `validators.py` (optional jinja meta check; size limits)

2) Default Templates & Presets
- Create 3 templates: comprehensive‑analysis, key‑insights, summary‑open‑questions
- Create presets: tones.yaml, roles.yaml, lengths.yaml (direct YAML dicts)

3) Public API Surface
- `src/prompting/__init__.py` re‑exports: list_templates, get_template, render_prompt, format_messages, list_presets

4) UI Integration
- Update `src/app.py` to use `list_templates()` and `format_messages()`
- Replace tone/role/length pickers to pull from presets YAML

5) Tests
- Add new unit + integration tests for prompting package
- Update E2E tests to validate template catalog and rendering

6) Delete Legacy
- Remove `src/prompts.py`
- Delete `tests/unit/prompts/test_prompts.py`
- Remove E2E test assertions referencing PREDEFINED_PROMPTS; replace with template registry assertions

7) Docs
- Update README “Choosing Prompts” to reference templates and presets
- Cross‑link ADR‑020 and this spec; add developer guide: how to add a template

## Acceptance Criteria

- All templates parse and validate at startup; `list_templates()` returns ≥3
- UI loads templates and renders with default context without errors
- Unit + integration + E2E tests pass without legacy prompt constants
- No references to `src/prompts.py` remain in repo
- Telemetry logs include `prompt.template_id` and `prompt.version`
- Zero custom Jinja environment code; rely on LlamaIndex `RichPromptTemplate`

## Risks & Mitigations

- Risk: Template variable drift between UI and templates → Strong validation, tests
- Risk: Over‑templating reduces flexibility → Keep simple presets and a small curated catalog
- Risk: Developer friction editing templates → Provide guidelines, examples, and lint
