"""Unit tests for conversation models in src.models.schemas."""

from __future__ import annotations

import importlib


def test_conversation_add_turn_and_context_window():  # type: ignore[no-untyped-def]
    models = importlib.import_module("src.models.schemas")
    ctx = models.ConversationContext(session_id="s1")
    t0 = models.ConversationTurn(id="0", role="user", content="hello world")
    t1 = models.ConversationTurn(id="1", role="assistant", content="hello world")
    t2 = models.ConversationTurn(id="2", role="user", content="hello world")
    t3 = models.ConversationTurn(id="3", role="assistant", content="hello world")
    t4 = models.ConversationTurn(id="4", role="user", content="hello world")
    ctx.add_turn(t0)
    ctx.add_turn(t1)
    ctx.add_turn(t2)
    ctx.add_turn(t3)
    ctx.add_turn(t4)
    assert ctx.total_tokens > 0
    # Get a small window
    window = ctx.get_context_window(max_tokens=10)
    # With simplified 2x-word count scoring, 'hello world' => 4 tokens per turn
    # So 2 turns fit in 10 tokens budget
    assert 1 <= len(window) <= 3
