"""Unit tests for prompting validators (SPEC-020)."""

from __future__ import annotations

from src.prompting.models import TemplateMeta, TemplateSpec
from src.prompting.validators import check_undeclared_variables, validate_meta


def test_validate_meta_ok() -> None:
    meta = TemplateMeta(
        id="t1",
        name="T1",
        description="desc",
        version=1,
    )
    validate_meta(meta)


def test_check_undeclared_variables_detects_missing() -> None:
    meta = TemplateMeta(id="t2", name="T2", description="", version=1)
    spec = TemplateSpec(meta=meta, body="Hello {{ missing_var }}")
    missing = check_undeclared_variables(spec)
    assert "missing_var" in missing
