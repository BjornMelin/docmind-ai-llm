"""Shared Qdrant exception groups for consistent handling."""

from __future__ import annotations

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

QDRANT_TRANSPORT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ResponseHandlingException,
    UnexpectedResponse,
    ConnectionError,
    TimeoutError,
    OSError,
)

QDRANT_SCHEMA_EXCEPTIONS: tuple[type[BaseException], ...] = (
    *QDRANT_TRANSPORT_EXCEPTIONS,
    ValueError,
    TypeError,
    AttributeError,
)

__all__ = [
    "QDRANT_SCHEMA_EXCEPTIONS",
    "QDRANT_TRANSPORT_EXCEPTIONS",
]
