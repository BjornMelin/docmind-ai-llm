"""Unit tests for prompting registry and renderer (SPEC-020)."""

from __future__ import annotations

import pytest

from src.prompting import format_messages, list_templates, render_prompt


def _get_test_template():
    """Helper function to get a test template, preferring comprehensive-analysis."""
    from src.prompting import registry as _registry

    _registry._catalog.cache_clear()
    templates = list_templates()
    if not templates:
        pytest.fail("No templates available for testing")

    # Try to find the comprehensive-analysis template, fallback to first available
    return next(
        (t for t in templates if t.id == "comprehensive-analysis"), templates[0]
    )


def test_render_prompt_and_messages() -> None:
    # Pick a known template id/name
    tpl = _get_test_template()

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


def test_key_insights_renders_role_and_tone_descriptions() -> None:
    text = render_prompt(
        "key-insights",
        {
            "context": "Example context",
            "role": {"description": "Provide custom actionable guidance."},
            "tone": {"description": "Use a concise tone."},
        },
    )

    assert "Provide custom actionable guidance." in text
    assert "Use a concise tone." in text


def test_render_raises_on_missing_required_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.prompting import registry as registry_module
    from src.prompting.models import TemplateMeta, TemplateSpec

    meta = TemplateMeta(
        id="stub-template",
        name="Stub",
        description="Stub description",
        required=["context"],
        defaults={},
    )
    spec = TemplateSpec(meta=meta, body="{context}")

    monkeypatch.setattr(registry_module, "get_template", lambda template_id: spec)

    ctx = {"tone": {"description": "Neutral"}, "role": {"description": "Assistant"}}
    with pytest.raises(KeyError, match="Missing required variables"):
        render_prompt(spec.meta.id, ctx)
