"""Tests for the release version parity gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_release_contract import validate_release_contract


def _write_contract(root: Path, *, manifest_version: str = "1.2.3") -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "docmind-ai-llm"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "docmind-ai-llm"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    (root / ".release-please-manifest.json").write_text(
        json.dumps({".": manifest_version}),
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [1.2.3] (2026-07-13)\n",
        encoding="utf-8",
    )


def test_release_contract_accepts_matching_versions(tmp_path: Path) -> None:
    _write_contract(tmp_path)

    assert validate_release_contract(tmp_path) == "1.2.3"


def test_release_contract_rejects_version_drift(tmp_path: Path) -> None:
    _write_contract(tmp_path, manifest_version="1.2.2")

    with pytest.raises(ValueError, match="release versions disagree"):
        validate_release_contract(tmp_path)
