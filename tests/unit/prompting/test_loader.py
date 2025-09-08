"""Unit tests for the prompting loader (SPEC-020).

Covers loading template specs from disk and preset YAML files.
"""

from __future__ import annotations

from src.prompting.loader import load_preset, load_templates


def test_load_templates_returns_specs() -> None:
    specs = load_templates()
    assert isinstance(specs, list)
    assert specs, "Expected at least one template spec on disk"
    ids = {s.meta.id for s in specs}
    assert "comprehensive-analysis" in ids


def test_load_presets_have_expected_keys() -> None:
    tones = load_preset("tones")
    roles = load_preset("roles")
    assert isinstance(tones, dict)
    assert isinstance(roles, dict)
    # Check a couple of defaults
    assert "professional" in tones
    assert "assistant" in roles
