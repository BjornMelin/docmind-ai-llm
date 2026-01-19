"""Wrapper module that re-exports the ingest adapter implementation.

This indirection allows tests that temporarily stub ``src.ui.ingest_adapter``
to be patched post-import with the real implementation when needed.
"""

from . import _ingest_adapter_impl as _impl

ingest_files = _impl.ingest_files
ingest_inputs = _impl.ingest_inputs
save_uploaded_file = _impl.save_uploaded_file

__all__ = ["ingest_files", "ingest_inputs", "save_uploaded_file"]
