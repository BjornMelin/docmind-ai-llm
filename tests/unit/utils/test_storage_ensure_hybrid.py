"""Tests for ensure_hybrid_collection behavior with fake client.

Ensures idempotent paths and defensive exception handling do not raise.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.utils.storage import ensure_hybrid_collection


class _ClientFake:
    def __init__(self, exists: bool = True, raise_on: str | None = None) -> None:
        self._exists = exists
        self._raise_on = raise_on
        self.created: dict[str, Any] | None = None
        self.updated = False

    def collection_exists(self, _name: str) -> bool:  # pragma: no cover - trivial
        if self._raise_on == "exists":
            raise ValueError("exists error")
        return self._exists

    def create_collection(self, **kwargs: Any) -> None:  # pragma: no cover - trivial
        if self._raise_on == "create":
            raise ValueError("create error")
        self.created = kwargs

    def update_collection(self, **_kwargs: Any) -> None:  # pragma: no cover - trivial
        if self._raise_on == "update":
            raise ValueError("update error")
        self.updated = True


@pytest.mark.unit
def test_ensure_hybrid_collection_idempotent() -> None:
    client = _ClientFake(exists=True)
    # Should not raise and not create
    ensure_hybrid_collection(client, "col")
    assert client.created is None


@pytest.mark.unit
@pytest.mark.parametrize("phase", ["exists", "create", "update"])
def test_ensure_hybrid_collection_defensive_no_raise(phase: str) -> None:
    client = _ClientFake(exists=False, raise_on=phase)
    # Should not raise even if client methods raise
    ensure_hybrid_collection(client, "col")
