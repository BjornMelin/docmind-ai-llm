"""DocMind AI - Local LLM for AI-Powered Document Analysis.

A privacy-focused document analysis system with hybrid search, knowledge graphs,
and multi-agent coordination for intelligent document processing.

Note: Module name intentionally kept as-is to match project name in pyproject.toml.
Ruff's N999 rule is temporarily disabled for this module.
"""  # noqa: N999

__version__ = "0.1.0"
__author__ = "Bjorn Melin"

# Make key components available at package level
from .config import settings

# Note: ADR-009 compliant processing available via:
# from src.processing import ResilientDocumentProcessor
# from src.cache import create_cache_manager

__all__ = [
    "settings",
]
