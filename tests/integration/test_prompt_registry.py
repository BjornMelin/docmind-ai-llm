"""Integration test for prompt registry (SPEC-020)."""

from __future__ import annotations

from src.prompting import list_templates, render_prompt


def test_prompt_registry_render_smoke():
    templates = list_templates()
    assert templates
    tpl = templates[0]
    ctx = {"context": "Integration smoke", "tone": {"description": "Neutral"}}
    text = render_prompt(tpl.id, ctx)
    assert isinstance(text, str)
    assert text.strip()
