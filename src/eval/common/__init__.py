"""Common evaluation utilities (determinism, sorting, mapping).

These helpers are intentionally small and dependency-light to keep
the evaluation harness deterministic and easy to maintain.
"""

from .determinism import set_determinism
from .mapping import build_doc_mapping, to_doc_id

__all__ = [
    "build_doc_mapping",
    "set_determinism",
    "to_doc_id",
]
