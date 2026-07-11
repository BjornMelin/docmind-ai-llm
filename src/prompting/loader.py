"""Template loader utilities.

Scans package resources for prompt files, parses YAML front matter,
and returns typed template specifications. Validation is minimal by design
(SPEC-020: KISS/DRY/YAGNI).
"""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Any

import yaml

from .models import TemplateMeta, TemplateSpec

_RESOURCE_ROOT = files("src.prompting").joinpath("templates")
_TPL_DIR: Traversable = _RESOURCE_ROOT.joinpath("prompts")
_PRESETS_DIR: Traversable = _RESOURCE_ROOT.joinpath("presets")

# Template body length threshold for auto-generating description
_MIN_BODY_LENGTH_FOR_DESCRIPTION = 40


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML front matter from a template body.

    Args:
        text: Full file content.

    Returns:
        Tuple of (front_matter_dict, body_str). Front matter may be empty.
    """
    if text.startswith("---\n"):
        try:
            _, fm, body = text.split("---\n", 2)
            data = yaml.safe_load(fm) or {}
            return data, body
        except ValueError:
            # No closing marker; treat entire file as body
            return {}, text
    return {}, text


def load_templates() -> list[TemplateSpec]:
    """Load all templates from disk.

    Returns:
        List of TemplateSpec objects for each `*.prompt.md` under templates/.
    """
    specs: list[TemplateSpec] = []
    if not _TPL_DIR.is_dir():
        return specs
    paths = sorted(
        (
            path
            for path in _TPL_DIR.iterdir()
            if path.is_file() and path.name.endswith(".prompt.md")
        ),
        key=lambda path: path.name,
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        fm, body = _split_front_matter(text)
        meta = TemplateMeta(
            id=str(fm.get("id") or path.name.removesuffix(".prompt.md")),
            name=str(
                fm.get("name")
                or path.name.removesuffix(".prompt.md").replace("-", " ").title()
            ),
            description=str(fm.get("description") or ""),
            tags=list(fm.get("tags") or []),
            required=list(fm.get("required") or []),
            defaults=dict(fm.get("defaults") or {}),
            version=int(fm.get("version") or 1),
        )
        # Basic body sanity
        if (
            len(body.strip()) < _MIN_BODY_LENGTH_FOR_DESCRIPTION
            and not meta.description
        ):
            meta.description = "Short template"
        specs.append(TemplateSpec(meta=meta, body=body))
    return specs


def load_preset(kind: str) -> dict[str, Any]:
    """Load a preset YAML file by kind.

    Args:
        kind: One of "tones", "roles", "lengths".

    Returns:
        Dictionary of preset entries.
    """
    path = _PRESETS_DIR.joinpath(f"{kind}.yaml")
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
