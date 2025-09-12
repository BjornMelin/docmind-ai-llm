"""Tests for prompting renderer helpers.

Covers compile_template, render_prompt, and format_messages.
"""

from __future__ import annotations

import pytest

from src.prompting.models import TemplateMeta, TemplateSpec
from src.prompting.renderer import compile_template, format_messages, render_prompt


@pytest.mark.unit
def test_renderer_text_and_messages() -> None:
    """Test template compilation, prompt rendering, and message formatting."""
    spec = TemplateSpec(
        meta=TemplateMeta(
            id="t",
            name="T",
            description="d",
            required=[],
            defaults={},
            version=1,
        ),
        body="Hello {{ name }}",
    )
    tpl = compile_template(spec)
    assert tpl is not None
    text = render_prompt(spec, {"name": "World"})
    assert "World" in text
    msgs = format_messages(spec, {"name": "World"})
    assert isinstance(msgs, list)
