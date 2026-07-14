#!/usr/bin/env python
"""Validate that every release-owned version source agrees."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

PACKAGE_NAME = "docmind-ai-llm"
VERSION_HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _locked_package_version(lock: dict[str, Any]) -> str:
    for package in lock.get("package", []):
        if package.get("name") == PACKAGE_NAME:
            version = package.get("version")
            if isinstance(version, str):
                return version
    raise ValueError(f"uv.lock has no {PACKAGE_NAME!r} package version")


def validate_release_contract(root: Path) -> str:
    """Return the canonical version or raise when release sources diverge."""
    pyproject = _read_toml(root / "pyproject.toml")
    lock = _read_toml(root / "uv.lock")
    manifest = json.loads(
        (root / ".release-please-manifest.json").read_text(encoding="utf-8")
    )
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")

    versions = {
        "pyproject.toml": pyproject.get("project", {}).get("version"),
        "uv.lock": _locked_package_version(lock),
        ".release-please-manifest.json": manifest.get("."),
    }
    invalid = {name: value for name, value in versions.items() if not value}
    if invalid:
        raise ValueError(f"missing release version values: {invalid}")

    unique = set(versions.values())
    if len(unique) != 1:
        details = ", ".join(f"{name}={value}" for name, value in versions.items())
        raise ValueError(f"release versions disagree: {details}")

    version = next(iter(unique))
    heading = VERSION_HEADING.search(changelog)
    if heading is None:
        raise ValueError("CHANGELOG.md has no released SemVer heading")
    if heading.group(1) != version:
        raise ValueError(
            "latest CHANGELOG.md release does not match metadata: "
            f"changelog={heading.group(1)}, metadata={version}"
        )
    return version


def main() -> int:
    """Validate the repository containing this script."""
    root = Path(__file__).resolve().parent.parent
    try:
        version = validate_release_contract(root)
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as exc:
        print(f"Release contract invalid: {exc}", file=sys.stderr)
        return 1
    print(f"Release contract valid for {PACKAGE_NAME} {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
