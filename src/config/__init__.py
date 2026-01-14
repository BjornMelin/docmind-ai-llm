"""Unified configuration interface for DocMind AI.

This module provides the primary configuration interface following ADR-024
implementing Task 2.2.2: LlamaIndex Settings Integration Pattern.

Key Features:
- Single source of truth via unified settings object
- Explicit LlamaIndex integration entrypoints (no import-time side effects)
- Environment variable support for all configuration
- Simplified import patterns across codebase

Usage:
    # Primary pattern (recommended for all code)
    from src.config import settings

    # LlamaIndex integration functions
    from src.config import setup_llamaindex, initialize_integrations
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .settings import settings

if TYPE_CHECKING:  # pragma: no cover
    from src.config.integrations import initialize_integrations, setup_llamaindex

__all__ = [
    "initialize_integrations",
    "settings",
    "setup_llamaindex",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "initialize_integrations": (".integrations", "initialize_integrations"),
    "setup_llamaindex": (".integrations", "setup_llamaindex"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve integration helpers to keep `import src.config` light."""
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Customize dir() to include all exports."""
    return sorted(set(list(globals()) + __all__))
