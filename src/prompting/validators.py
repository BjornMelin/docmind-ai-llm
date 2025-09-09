"""Template validators (minimal, optional checks).

Implements small helpers for validating template metadata and optionally
running static checks on Jinja variables per SPEC-020.
"""

from __future__ import annotations

from jinja2 import Environment
from jinja2 import meta as jinja_meta

from .models import TemplateMeta, TemplateSpec


def validate_meta(meta: TemplateMeta) -> None:
    """Validate that required metadata fields are present and sane.

    Args:
        meta: Template metadata to validate.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if not meta.id or not meta.name:
        raise ValueError("Template id and name are required")
    if meta.version < 1:
        raise ValueError("Template version must be >= 1")


def check_undeclared_variables(
    spec: TemplateSpec, autoescape: bool = False
) -> set[str]:
    """Return undeclared Jinja variables in the template body.

    Useful for early detection of missing variables. This is an optional
    static check; rendering still enforces required variables at runtime.

    Args:
        spec: Template specification with body content.
        autoescape: Enable HTML/XML autoescaping. Set True only if templates
            produce HTML or XML. Default False for general purpose templates
            like LLM prompts which are plain text.
    """
    env = Environment(autoescape=autoescape)  # noqa: S701 Plain text, not HTML
    ast = env.parse(spec.body)
    return set(jinja_meta.find_undeclared_variables(ast))
