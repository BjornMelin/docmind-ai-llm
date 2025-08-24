"""DocMind AI - Local LLM for AI-Powered Document Analysis.

A privacy-focused document analysis system with hybrid search, knowledge graphs,
and multi-agent coordination for intelligent document processing.

Note: Module name intentionally kept as-is to match project name in pyproject.toml.
Ruff's N999 rule is temporarily disabled for this module.
"""  # noqa: N999

__version__ = "0.1.0"
__author__ = "Bjorn Melin"

# Make key components available at package level
from .config.settings import settings
from .retrieval.integration import create_hybrid_retriever, create_index_async
from .utils.document import load_documents_unstructured

__all__ = [
    "settings",
    "load_documents_unstructured",
    "create_index_async",
    "create_hybrid_retriever",
]
