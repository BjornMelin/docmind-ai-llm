"""Query optimization modules for DocMind AI.

This module implements DSPy-based query optimization functionality per ADR-018
for enhanced retrieval performance.
"""

from .dspy_optimizer import DSPyQueryOptimizer

__all__ = ["DSPyQueryOptimizer"]
