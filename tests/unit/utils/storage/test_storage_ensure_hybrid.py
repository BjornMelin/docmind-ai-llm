"""Tests for canonical ensure_hybrid_collection behavior with a fake client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models as qmodels

from src.utils.storage import (
    canonical_text_collection_metadata,
    ensure_hybrid_collection,
)


class _ClientFake:
    """Lightweight Qdrant client stub used to simulate collection states."""

    def __init__(self, exists: bool = True, raise_on: str | None = None) -> None:
        self._exists = exists
        self._raise_on = raise_on
        self.created: dict[str, Any] | None = None

    def collection_exists(self, _name: str) -> bool:  # pragma: no cover - trivial
        if self._raise_on == "exists":
            raise ValueError("exists error")
        return self._exists

    def create_collection(self, **kwargs: Any) -> None:  # pragma: no cover - trivial
        if self._raise_on == "create":
            raise ValueError("create error")
        self.created = kwargs

    def get_collection(self, _name: str) -> SimpleNamespace:
        if self._raise_on == "get":
            raise ValueError("get error")
        params = SimpleNamespace(
            vectors={
                "text-dense": SimpleNamespace(
                    size=1024,
                    distance=qmodels.Distance.COSINE,
                )
            },
            sparse_vectors={
                "text-sparse": SimpleNamespace(modifier=qmodels.Modifier.IDF)
            },
        )
        return SimpleNamespace(
            config=SimpleNamespace(
                params=params,
                metadata=canonical_text_collection_metadata(
                    dense_dim=1024,
                    sparse_enabled=True,
                ),
            ),
            points_count=1,
        )


@pytest.mark.unit
def test_ensure_hybrid_collection_idempotent() -> None:
    """Verify that ensure_hybrid_collection is a no-op when collection exists."""
    client = _ClientFake(exists=True)
    result = ensure_hybrid_collection(client, "col", sparse_enabled=True)

    assert result.compatible is True
    assert result.action == "unchanged"
    assert client.created is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("phase", "exists", "reason"),
    [
        ("exists", False, "compatibility_check_failed"),
        ("get", True, "compatibility_check_failed"),
        ("create", False, "collection_create_failed"),
    ],
)
def test_ensure_hybrid_collection_reports_client_errors(
    phase: str,
    exists: bool,
    reason: str,
) -> None:
    """Client errors return explicit incompatibility evidence without raising."""
    client = _ClientFake(exists=exists, raise_on=phase)

    result = ensure_hybrid_collection(client, "col", sparse_enabled=True)

    assert result.compatible is False
    assert result.action == "error"
    assert result.reason == reason
