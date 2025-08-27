"""DocMind AI Cache Module.

Caching implementation:
- SQLite-based document processing cache
- Single library dependency (LlamaIndex)
- Multi-agent coordination via WAL mode

"""

from src.cache.simple_cache import SimpleCache

__all__ = [
    "SimpleCache",
]
