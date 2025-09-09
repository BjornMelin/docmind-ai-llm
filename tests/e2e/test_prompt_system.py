"""E2E smoke for prompt system (SPEC-020)."""

from __future__ import annotations

import pytest

from src.prompting import list_templates


@pytest.mark.e2e
def test_prompt_catalog_present():
    templates = list_templates()
    assert isinstance(templates, list)
    assert templates
    names = [t.name for t in templates]
    assert any("Comprehensive" in n for n in names)
