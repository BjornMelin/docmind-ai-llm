"""Template loader utilities.

Scans the templates/ directory for prompt files, parses YAML front matter,
and returns typed template specifications. Validation is minimal by design
(SPEC-020: KISS/DRY/YAGNI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import TemplateMeta, TemplateSpec

_ROOT = Path(__file__).resolve().parents[2]
_TPL_DIR = _ROOT / "templates" / "prompts"
_PRESETS_DIR = _ROOT / "templates" / "presets"

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
    if not _TPL_DIR.exists():
        return specs
    for path in sorted(_TPL_DIR.glob("*.prompt.md")):
        text = path.read_text(encoding="utf-8")
        fm, body = _split_front_matter(text)
        meta = TemplateMeta(
            id=str(fm.get("id") or path.stem),
            name=str(fm.get("name") or path.stem.replace("-", " ").title()),
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
    path = _PRESETS_DIR / f"{kind}.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
