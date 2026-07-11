"""Typed failures for the document parsing boundary."""

from __future__ import annotations

from pathlib import Path


class DocumentParseError(RuntimeError):
    """Report a format-aware parsing failure without exposing document contents."""

    def __init__(
        self,
        path: Path,
        *,
        stage: str,
        reason: str,
        cause: BaseException | None = None,
        cause_type: str | None = None,
    ) -> None:
        """Initialize a safe parsing error.

        Args:
            path: Source document path. Only its basename and suffix are retained.
            stage: Stable parsing stage identifier.
            reason: Stable machine-readable failure reason.
            cause: Optional originating exception. Only its type is retained.
            cause_type: Safe cause type received across a process boundary.
        """
        source = Path(path)
        self.source_filename = source.name
        self.source_suffix = source.suffix.lower()
        self.stage = stage
        self.reason = reason
        if cause is not None and cause_type is not None:
            raise ValueError("provide cause or cause_type, not both")
        self.cause_type = type(cause).__name__ if cause is not None else cause_type
        format_name = self.source_suffix or "<unknown>"
        cause_part = f", cause_type={self.cause_type}" if self.cause_type else ""
        super().__init__(
            f"Document parsing failed for {self.source_filename} "
            f"(format={format_name}, stage={stage}, reason={reason}{cause_part})"
        )


__all__ = ["DocumentParseError"]
