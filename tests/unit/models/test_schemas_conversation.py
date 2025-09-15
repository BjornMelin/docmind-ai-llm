"""Unit tests for conversation models in src.models.schemas."""

from __future__ import annotations

import importlib


def test_conversation_add_turn_and_context_window():  # type: ignore[no-untyped-def]
    models = importlib.import_module("src.models.schemas")
    ctx = models.ConversationContext(session_id="s1")
    # Add several turns
    for i in range(5):
        t = models.ConversationTurn(id=str(i), role="user", content="hello world")
        ctx.add_turn(t)
    assert ctx.total_tokens > 0
    # Get a small window
    window = ctx.get_context_window(max_tokens=10)
    # With simplified 2x-word count scoring, 'hello world' => 4 tokens per turn
    # So 2 turns fit in 10 tokens budget
    assert 1 <= len(window) <= 3
