"""Unit tests for prompting registry and renderer (SPEC-020)."""

from __future__ import annotations

from src.prompting import (
    format_messages,
    list_templates,
    render_prompt,
)


def test_render_prompt_and_messages() -> None:
    # Pick a known template id/name
    templates = list_templates()
    assert templates
    tpl = next((t for t in templates if t.id == "comprehensive-analysis"), templates[0])

    ctx = {
        "context": "Example context",
        "tone": {"description": "Use a neutral tone."},
        "role": {"description": "Act as a helpful assistant."},
    }

    text = render_prompt(tpl.id, ctx)
    assert isinstance(text, str)
    assert text.strip()

    msgs = format_messages(tpl.id, ctx)
    assert isinstance(msgs, list)
    assert msgs, "Expected at least one chat message"
