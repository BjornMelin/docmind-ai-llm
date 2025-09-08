"""Prompt rendering using LlamaIndex RichPromptTemplate.

Provides thin helpers to render text prompts or chat messages from loaded
`TemplateSpec` objects. Keeps zero custom Jinja environment state per SPEC-020.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.prompts import RichPromptTemplate

from .models import TemplateSpec


def compile_template(spec: TemplateSpec) -> RichPromptTemplate:
    """Compile a `TemplateSpec` body into a RichPromptTemplate.

    Args:
        spec: The template specification with body content.

    Returns:
        A compiled `RichPromptTemplate` ready for rendering.
    """
    return RichPromptTemplate(spec.body)


def render_prompt(spec: TemplateSpec, context: dict[str, Any]) -> str:
    """Render a text prompt from a `TemplateSpec`.

    Args:
        spec: Template specification.
        context: Variables to provide to the template.

    Returns:
        Rendered text prompt.
    """
    return compile_template(spec).format(**context)


def format_messages(spec: TemplateSpec, context: dict[str, Any]) -> list[Any]:
    """Render chat messages from a `TemplateSpec`.

    Args:
        spec: Template specification.
        context: Variables to provide to the template.

    Returns:
        A list of chat message objects suitable for LLM chat APIs.
    """
    return compile_template(spec).format_messages(**context)
