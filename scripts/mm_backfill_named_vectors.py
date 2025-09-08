"""Stub for future named-vectors backfill.

This script will iterate existing Qdrant collections and ensure `text-dense`
and `text-sparse` named vectors are present (migrate from legacy schemas).

Current status: no-op placeholder to satisfy SPEC-004 migration notes.
"""

from __future__ import annotations


def main() -> None:
    """Entry point for the backfill stub (no-op)."""
    print("named-vectors backfill stub (no-op)")


if __name__ == "__main__":
    main()
