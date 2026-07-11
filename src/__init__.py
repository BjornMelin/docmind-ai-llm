"""DocMind AI - Local LLM for AI-Powered Document Analysis.

A privacy-focused document analysis system with hybrid search, knowledge graphs,
and multi-agent coordination for intelligent document processing.

The package keeps import-time work minimal so release and health checks can
import it without initializing application settings or external integrations.
"""

from src.version import get_version

__version__ = get_version()
__author__ = "Bjorn Melin"

__all__ = [
    "__version__",
]
