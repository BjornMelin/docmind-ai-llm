"""Unit tests for prompting validators.

Covers metadata validation errors and detection of undeclared variables in
templates using Jinja environment inspection.
"""

from __future__ import annotations

import pytest

from src.prompting.models import TemplateMeta, TemplateSpec
from src.prompting.validators import check_undeclared_variables, validate_meta


@pytest.mark.unit
def test_validate_meta_success() -> None:
    """Test that valid metadata passes validation without raising exceptions."""
    meta = TemplateMeta(id="greet", name="Greeting", version=1)
    validate_meta(meta)  # should not raise


@pytest.mark.unit
def test_validate_meta_version_error() -> None:
    """Test that metadata with invalid version raises ValueError."""
    meta = TemplateMeta(id="x", name="y", version=0, description="d")
    with pytest.raises(ValueError, match="version"):
        validate_meta(meta)


@pytest.mark.unit
def test_empty_fields_raise_in_validate_meta() -> None:
    """Test that empty id or name is rejected by validate_meta."""
    meta1 = TemplateMeta(id="", name="X", version=1, description="d")
    with pytest.raises(ValueError, match="id and name"):
        validate_meta(meta1)
    meta2 = TemplateMeta(id="a", name="", version=1, description="d")
    with pytest.raises(ValueError, match="id and name"):
        validate_meta(meta2)


@pytest.mark.unit
def test_check_undeclared_variables_detects_placeholders() -> None:
    """Test that undeclared variables are detected in template bodies."""
    spec = TemplateSpec(
        meta=TemplateMeta(id="greet", name="Greeting", version=1, description="d"),
        body="Hello {{ name }}, today is {{ day }}.",
    )
    vars_ = check_undeclared_variables(spec)
    assert {"name", "day"}.issubset(vars_)
