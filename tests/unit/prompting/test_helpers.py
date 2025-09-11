"""Tests for src.prompting.helpers.build_prompt_context."""

import pytest

from src.prompting.helpers import build_prompt_context


def test_build_prompt_context_success(monkeypatch):
    """Happy path: context resolves presets and renders via registry."""
    resources = {
        "tones": {"formal": {"style": "formal"}},
        "roles": {"analyst": {"role": "analyst"}},
    }

    # Minimal template registry shim via monkeypatch on render_prompt path
    from src.prompting import helpers as _helpers

    called = {}

    def _fake_render_prompt(tid, ctx):
        called["tid"] = tid
        called["ctx"] = ctx
        return f"T:{tid}|tone={ctx['tone'].get('style')}|role={ctx['role'].get('role')}"

    monkeypatch.setattr(_helpers, "render_prompt", _fake_render_prompt)

    out = build_prompt_context(
        template_id="tmpl-1",
        tone_selection="formal",
        role_selection="analyst",
        resources=resources,
    )
    assert out.startswith("T:tmpl-1|")
    assert called["tid"] == "tmpl-1"
    assert called["ctx"]["tone"]["style"] == "formal"
    assert called["ctx"]["role"]["role"] == "analyst"


def test_build_prompt_context_missing_vars(monkeypatch):
    """Registry raises KeyError; helper converts to RuntimeError."""
    # Simulate registry raising KeyError (missing required ctx variables)
    from src.prompting import helpers as _helpers

    def _raise_keyerror(*_args, **_kwargs):
        raise KeyError("missing var")

    monkeypatch.setattr(_helpers, "render_prompt", _raise_keyerror)

    with pytest.raises(RuntimeError) as ei:
        _ = build_prompt_context(
            template_id="tmpl-err",
            tone_selection="any",
            role_selection="any",
            resources={},
        )
    assert "Template rendering failed" in str(ei.value)
