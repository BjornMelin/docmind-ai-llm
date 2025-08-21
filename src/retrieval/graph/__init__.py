"""Graph-based retrieval modules for DocMind AI.

This module implements PropertyGraphIndex functionality per ADR-019
for relationship-based document analysis.
"""

from .property_graph import PropertyGraphIndex

__all__ = ["PropertyGraphIndex"]
