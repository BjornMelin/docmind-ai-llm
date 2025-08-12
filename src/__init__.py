"""DocMind AI - Local LLM for AI-Powered Document Analysis.

A privacy-focused document analysis system with hybrid search, knowledge graphs,
and multi-agent coordination for intelligent document processing.
"""

__version__ = "0.1.0"
__author__ = "Bjorn Melin"

# Make key components available at package level
from src.models.core import settings
from src.utils.document import load_documents_unstructured
from src.utils.embedding import create_index_async, create_hybrid_retriever

__all__ = [
    "settings",
    "load_documents_unstructured", 
    "create_index_async",
    "create_hybrid_retriever",
]
