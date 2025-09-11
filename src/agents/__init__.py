"""Agents package.

Keeps package initialization lightweight to avoid heavy imports during
module discovery and Streamlit AppTest. Submodules (e.g., ``coordinator``,
``tools``) should be imported explicitly by consumers.
"""

__all__ = []
