"""Additional tests for prompting.loader: front matter and auto-description."""

from __future__ import annotations

import importlib

import pytest
import yaml


@pytest.mark.unit
def test_split_front_matter_cases(tmp_path):
    mod = importlib.import_module("src.prompting.loader")

    # No front matter
    fm, body = mod._split_front_matter("hello")  # pylint: disable=protected-access
    assert fm == {}
    assert body == "hello"

    # Missing closing marker
    fm, body = mod._split_front_matter("---\nkey: v without closing")  # pylint: disable=protected-access
    assert fm == {}
    assert body.startswith("---\nkey")

    # Valid front matter, no description with short body → auto description
    d = tmp_path / "t.prompt.md"
    fm_data = {"id": "x", "name": "Name", "tags": ["a"], "defaults": {"k": 1}}
    content = "---\n" + yaml.safe_dump(fm_data) + "---\nshort body"
    d.write_text(content, encoding="utf-8")

    # Point loader to tmp dir and load
    mod._TPL_DIR = tmp_path  # type: ignore[attr-defined]
    specs = mod.load_templates()
    assert len(specs) == 1
    sp = specs[0]
    assert sp.meta.id == "x"
    assert sp.meta.name == "Name"
    assert sp.meta.tags == ["a"]
    assert sp.meta.defaults == {"k": 1}
    assert sp.meta.description == "Short template"


@pytest.mark.unit
def test_load_preset_known_and_unknown(tmp_path, monkeypatch):
    mod = importlib.import_module("src.prompting.loader")

    mod._PRESETS_DIR = tmp_path  # type: ignore[attr-defined]
    (tmp_path / "tones.yaml").write_text("professional: true\n", encoding="utf-8")

    tones = mod.load_preset("tones")
    assert tones.get("professional") is True
    assert mod.load_preset("missing-kind") == {}
