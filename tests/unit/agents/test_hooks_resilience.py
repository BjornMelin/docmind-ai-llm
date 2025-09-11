"""Tests for LangGraph pre/post hook resilience in MultiAgentCoordinator.

Ensures hook exceptions are caught and state is annotated rather than crashing.
"""

from __future__ import annotations

from typing import Any

from src.agents.coordinator import MultiAgentCoordinator


def test_pre_post_hooks_resilience(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Verify that hook exceptions do not crash and set observability flags.

    The test monkeypatches the context manager's token estimator to raise in the
    pre-model hook path and asserts the state is returned with ``hook_error`` and
    ``hook_name`` set.
    """
    coord = MultiAgentCoordinator()
    pre_hook = coord._create_pre_model_hook()  # pylint: disable=protected-access
    post_hook = coord._create_post_model_hook()  # pylint: disable=protected-access

    # Make the context manager raise when estimating tokens
    def _boom(_messages: list[Any]) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(coord.context_manager, "estimate_tokens", _boom)

    state = {"messages": ["hello"], "output_mode": "structured"}
    out_pre = pre_hook(state.copy())
    assert out_pre.get("hook_error") is True
    assert out_pre.get("hook_name") == "pre_model_hook"

    # Ensure post hook exception path also sets flags
    # (simulate by raising in calculate_kv_cache_usage)
    def _boom_kv(_state: dict) -> float:
        raise RuntimeError("boom-kv")

    monkeypatch.setattr(coord.context_manager, "calculate_kv_cache_usage", _boom_kv)
    out_post = post_hook(state.copy())
    assert out_post.get("hook_error") is True
    assert out_post.get("hook_name") == "post_model_hook"
