"""Template registry and public API.

Lazy-loads templates from disk, compiles them on first use, and exposes a
minimal API to list templates, get metadata, and render outputs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .loader import load_preset, load_templates
from .models import TemplateMeta, TemplateSpec
from .renderer import format_messages as _format_msgs
from .renderer import render_prompt as _render_text
from .validators import validate_meta


@lru_cache(maxsize=1)
def _catalog() -> dict[str, TemplateSpec]:
    specs = load_templates()
    out: dict[str, TemplateSpec] = {}
    for spec in specs:
        validate_meta(spec.meta)
        out[spec.meta.id] = spec
    return out


def list_templates() -> list[TemplateMeta]:
    """Return a list of available template metadata entries."""
    return [spec.meta for spec in _catalog().values()]


def get_template(template_id: str) -> TemplateSpec:
    """Return the TemplateSpec for a given template id.

    Raises KeyError if not found.
    """
    return _catalog()[template_id]


def render_prompt(template_id: str, context: dict[str, Any]) -> str:
    """Render a text prompt for a given template id and context."""
    spec = get_template(template_id)
    # Merge defaults with provided context
    ctx = {**spec.meta.defaults, **context}
    # Optionally check required
    if missing := [v for v in spec.meta.required if v not in ctx]:
        raise KeyError(f"Missing required variables: {missing}")
    return _render_text(spec, ctx)


def format_messages(template_id: str, context: dict[str, Any]) -> list[Any]:
    """Render chat messages for a given template id and context."""
    spec = get_template(template_id)
    ctx = {**spec.meta.defaults, **context}
    if missing := [v for v in spec.meta.required if v not in ctx]:
        raise KeyError(f"Missing required variables: {missing}")
    return _format_msgs(spec, ctx)


def list_presets(kind: str) -> dict[str, Any]:
    """Return presets for a given kind (tones|roles|lengths)."""
    return load_preset(kind)
