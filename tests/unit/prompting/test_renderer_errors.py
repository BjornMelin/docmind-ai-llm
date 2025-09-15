"""Tests for prompting.renderer error paths and message formatting.

Uses a stubbed RichPromptTemplate to avoid importing heavy libraries and to
control error behavior deterministically.
"""

from __future__ import annotations

from typing import Any

import pytest


class _FakeRichPromptTemplate:
    """Minimal stub for RichPromptTemplate.

    - ``format`` proxies to ``str.format`` on the body and raises KeyError on
      missing variables.
    - ``format_messages`` returns two message-like dicts for tests that assert
      multi-message structures.
    """

    def __init__(self, body: str) -> None:
        self._body = body

    def format(self, **context: Any) -> str:
        return self._body.format(**context)

    def format_messages(self, **context: Any):
        # Simulate two chat messages derived from context.
        sys_txt = context.get("s", "")
        usr_txt = context.get("u", "")
        return [
            {"role": "system", "content": f"System: {sys_txt}"},
            {"role": "user", "content": f"User: {usr_txt}"},
        ]


@pytest.mark.unit
def test_render_prompt_missing_variable_raises(monkeypatch) -> None:
    """Missing required variable causes a clean exception from renderer."""
    import importlib

    rnd = importlib.import_module("src.prompting.renderer")
    models = importlib.import_module("src.prompting.models")

    # Patch RichPromptTemplate in the renderer module
    monkeypatch.setattr(rnd, "RichPromptTemplate", _FakeRichPromptTemplate)

    spec = models.TemplateSpec(
        meta=models.TemplateMeta(
            id="t1",
            name="n",
            description="d",
            tags=[],
            required=["name"],
            defaults={},
            version=1,
        ),
        body="Hello {name}",
    )

    with pytest.raises(KeyError):
        rnd.render_prompt(spec, context={})


@pytest.mark.unit
def test_format_messages_two_messages(monkeypatch) -> None:
    """format_messages returns at least two messages for System/User bodies."""
    import importlib

    rnd = importlib.import_module("src.prompting.renderer")
    models = importlib.import_module("src.prompting.models")

    monkeypatch.setattr(rnd, "RichPromptTemplate", _FakeRichPromptTemplate)

    spec = models.TemplateSpec(
        meta=models.TemplateMeta(
            id="t2",
            name="n",
            description="d",
            tags=[],
            required=[],
            defaults={},
            version=1,
        ),
        body="System: {s}\nUser: {u}",
    )

    msgs = rnd.format_messages(spec, {"s": "sys", "u": "user"})
    assert isinstance(msgs, list)
    assert len(msgs) >= 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
