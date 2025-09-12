"""Tests for prompting helpers and registry behavior.

Focus on required variable enforcement, defaults merge, and error normalization.
"""

from __future__ import annotations

import pytest

import src.prompting.registry as reg
from src.prompting.helpers import build_prompt_context
from src.prompting.models import TemplateMeta, TemplateSpec


@pytest.mark.unit
def test_registry_render_prompt_required_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = TemplateMeta(
        id="t1",
        name="T",
        description="d",
        required=["who"],
        defaults={"context": "ctx"},
        version=1,
    )
    spec = TemplateSpec(meta=meta, body="Hello {{ who }}")

    # Monkeypatch catalog to our single spec
    monkeypatch.setattr(reg, "_catalog", lambda: {"t1": spec}, raising=True)

    with pytest.raises(KeyError):
        reg.render_prompt("t1", {"x": 1})

    out = reg.render_prompt("t1", {"who": "World"})
    assert "World" in out


@pytest.mark.unit
def test_helpers_build_prompt_context_normalizes_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Patch render_prompt to raise KeyError and assert normalization to RuntimeError
    def _raise_keyerror(*_a, **_k):  # pragma: no cover - trivial
        raise KeyError("missing")

    import src.prompting.helpers as helpers

    monkeypatch.setattr(helpers, "render_prompt", _raise_keyerror, raising=True)
    with pytest.raises(RuntimeError, match="Template rendering failed"):
        build_prompt_context(
            template_id="t1",
            tone_selection="friendly",
            role_selection="instructor",
            resources={"tones": {"friendly": {}}, "roles": {"instructor": {}}},
        )

    # Patch to echo back a simple string to prove path
    monkeypatch.setattr(
        helpers, "render_prompt", lambda _tid, ctx: ctx["context"], raising=True
    )
    out = build_prompt_context(
        template_id="t1",
        tone_selection="friendly",
        role_selection="instructor",
        resources={"tones": {"friendly": {}}, "roles": {"instructor": {}}},
    )
    assert isinstance(out, str)
