"""Wrapper module that re-exports the ingest adapter implementation.

This indirection allows tests that temporarily stub ``src.ui.ingest_adapter``
to be patched post-import with the real implementation when needed.
"""

from ._ingest_adapter_impl import *  # noqa: F403

__all__ = [name for name in globals() if not name.startswith("_")]
