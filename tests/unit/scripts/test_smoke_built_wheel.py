"""Tests for release-wheel contents and metadata validation."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts import smoke_built_wheel as smoke

pytestmark = pytest.mark.unit


def _write_wheel(
    path: Path,
    *,
    dependencies: set[str],
    extras: set[str] | None = None,
    extra_files: set[str] | None = None,
) -> None:
    metadata = [
        "Metadata-Version: 2.4",
        "Name: docmind_ai_llm",
        "Version: 0.9.0",
        *(f"Requires-Dist: {name}" for name in sorted(dependencies)),
        *(f"Provides-Extra: {name}" for name in sorted(extras or set())),
        "",
    ]
    with zipfile.ZipFile(path, "w") as archive:
        for name in smoke._REQUIRED_WHEEL_FILES:
            archive.writestr(name, "placeholder")
        for name in extra_files or set():
            archive.writestr(name, "stale")
        archive.writestr(
            "docmind_ai_llm-0.9.0.dist-info/METADATA",
            "\n".join(metadata),
        )


def test_validate_wheel_accepts_canonical_dependency_contract(tmp_path: Path) -> None:
    wheel = tmp_path / "docmind_ai_llm-0.9.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        dependencies=set(smoke._REQUIRED_WHEEL_DEPENDENCIES),
        extras={"gpu"},
    )

    smoke.validate_wheel_contents(wheel)


def test_validate_wheel_rejects_missing_required_dependency(tmp_path: Path) -> None:
    wheel = tmp_path / "docmind_ai_llm-0.9.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        dependencies=smoke._REQUIRED_WHEEL_DEPENDENCIES.difference({"torch"}),
    )

    with pytest.raises(SystemExit, match="missing required dependencies: torch"):
        smoke.validate_wheel_contents(wheel)


@pytest.mark.parametrize(
    "dependency",
    sorted(smoke._FORBIDDEN_WHEEL_DEPENDENCIES),
)
def test_validate_wheel_rejects_forbidden_dependencies(
    tmp_path: Path,
    dependency: str,
) -> None:
    wheel = tmp_path / "docmind_ai_llm-0.9.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        dependencies={*smoke._REQUIRED_WHEEL_DEPENDENCIES, dependency},
    )

    with pytest.raises(SystemExit, match="forbidden dependencies"):
        smoke.validate_wheel_contents(wheel)


@pytest.mark.parametrize("extra", sorted(smoke._FORBIDDEN_WHEEL_EXTRAS))
def test_validate_wheel_rejects_removed_extra(tmp_path: Path, extra: str) -> None:
    wheel = tmp_path / "docmind_ai_llm-0.9.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        dependencies=set(smoke._REQUIRED_WHEEL_DEPENDENCIES),
        extras={extra},
    )

    with pytest.raises(SystemExit, match=f"forbidden extras: {extra}"):
        smoke.validate_wheel_contents(wheel)


def test_validate_wheel_rejects_stale_build_tree_file(tmp_path: Path) -> None:
    wheel = tmp_path / "docmind_ai_llm-0.9.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        dependencies=set(smoke._REQUIRED_WHEEL_DEPENDENCIES),
        extra_files={"src/retrieval/deleted_module.py"},
    )

    with pytest.raises(SystemExit, match="files absent from the source tree"):
        smoke.validate_wheel_contents(wheel)
