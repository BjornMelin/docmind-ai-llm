"""Prompting helper utilities (pure functions, no UI side-effects).

This module contains small, reusable helpers for building prompt contexts
and rendering templates. It intentionally avoids Streamlit calls and telemetry
side-effects; callers decide how to handle UI and logging.
"""

from __future__ import annotations

from typing import Any

from .registry import render_prompt


def build_prompt_context(
    *,
    template_id: str,
    tone_selection: str,
    role_selection: str,
    resources: dict[str, Any],
) -> str:
    """Build a prompt context and render it using the template registry.

    Args:
        template_id: ID of the template to render.
        tone_selection: Key into resources['tones'] for tone preset.
        role_selection: Key into resources['roles'] for role preset.
        resources: Mapping that may contain 'tones', 'roles', and 'templates'.

    Returns:
        Rendered prompt string.

    Raises:
        RuntimeError: When required variables are missing for the template.
    """
    tones = resources.get("tones", {})
    roles = resources.get("roles", {})

    # Resolve selected presets
    tone = tones.get(tone_selection, {})
    role = roles.get(role_selection, {})

    ctx = {
        "context": "Docs indexed",  # caller should override with real context
        "tone": tone,
        "role": role,
    }

    try:
        return render_prompt(template_id, ctx)
    except KeyError as exc:  # normalize to RuntimeError for callers
        raise RuntimeError(f"Template rendering failed: {exc}") from exc


__all__ = ["build_prompt_context"]
