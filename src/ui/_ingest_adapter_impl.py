"""UI ingestion adapter placeholder during ingestion refactor.

Legacy ingestion, analytics, and GraphRAG wiring have been removed. The
functions in this module now raise ``NotImplementedError`` until the new
LlamaIndex-first ingestion pipeline is implemented in later phases.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def ingest_files(files: Sequence[Any], enable_graphrag: bool = False) -> dict[str, Any]:
    """Placeholder ingestion entrypoint.

    Args:
        files: Uploaded file-like objects.
        enable_graphrag: GraphRAG flag (unused).

    Raises:
        NotImplementedError: Always raised until the new pipeline lands.
    """
    raise NotImplementedError("UI ingestion adapter removed pending pipeline rebuild.")


def save_uploaded_file(_file: Any) -> Path:
    """Placeholder file persistence helper.

    Raises:
        NotImplementedError: Always raised until the new pipeline lands.
    """
    raise NotImplementedError(
        "File persistence helper removed pending pipeline rebuild."
    )
