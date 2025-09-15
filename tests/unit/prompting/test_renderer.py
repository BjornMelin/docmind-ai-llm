"""Unit tests for prompting.renderer helpers using TemplateSpec.

Validates compile_template, render_prompt, and format_messages with a simple
template body.
"""

from __future__ import annotations

import importlib


def _make_spec(body: str):  # type: ignore[no-untyped-def]
    models = importlib.import_module("src.prompting.models")
    meta = models.TemplateMeta(
        id="t1",
        name="test",
        description="d",
        tags=[],
        required=[],
        defaults={},
        version=1,
    )
    return models.TemplateSpec(meta=meta, body=body)


def test_compile_and_render_prompt_text():  # type: ignore[no-untyped-def]
    rnd = importlib.import_module("src.prompting.renderer")
    spec = _make_spec("Hello {{ name }}")
    out = rnd.render_prompt(spec, {"name": "DocMind"})
    assert out == "Hello DocMind"


def test_format_messages_returns_list():  # type: ignore[no-untyped-def]
    rnd = importlib.import_module("src.prompting.renderer")
    spec = _make_spec("System: {{ s }}\nUser: {{ u }}")
    msgs = rnd.format_messages(spec, {"s": "sys", "u": "user"})
    assert isinstance(msgs, list)
    assert msgs, "Messages should not be empty"
