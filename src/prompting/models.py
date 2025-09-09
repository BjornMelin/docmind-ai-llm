"""Prompt template models.

Defines Pydantic models for template metadata/specification used by the
file-based prompt system (SPEC-020). These models describe a template's
front matter and body content.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TemplateMeta(BaseModel):
    """Metadata for a single prompt template.

    Attributes:
        id: Unique identifier used to reference the template programmatically.
        name: Human-friendly template name.
        description: Short summary of the template's intent.
        tags: Optional tags for categorization and filtering.
        required: Variables that must be provided when rendering.
        defaults: Default variable values applied when rendering.
        version: Simple integer version for telemetry and evolution.
    """

    id: str = Field(description="Unique template identifier")
    name: str = Field(description="Human-readable template name")
    description: str = Field(description="Template description")
    tags: list[str] = Field(default_factory=list, description="Tags")
    required: list[str] = Field(default_factory=list, description="Required vars")
    defaults: dict[str, Any] = Field(default_factory=dict, description="Defaults")
    version: int = Field(default=1, description="Template version")


class TemplateSpec(BaseModel):
    """Complete prompt template including metadata and body."""

    meta: TemplateMeta
    body: str = Field(description="Jinja/RichPromptTemplate body content")
