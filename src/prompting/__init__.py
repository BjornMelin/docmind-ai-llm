"""Prompting public API (SPEC-020).

Minimal, library-first wrapper around file-based templates rendered via
LlamaIndex `RichPromptTemplate`.
"""

from .models import TemplateMeta, TemplateSpec
from .registry import (
    format_messages,
    get_template,
    list_presets,
    list_templates,
    render_prompt,
)

__all__ = [
    "TemplateMeta",
    "TemplateSpec",
    "format_messages",
    "get_template",
    "list_presets",
    "list_templates",
    "render_prompt",
]
