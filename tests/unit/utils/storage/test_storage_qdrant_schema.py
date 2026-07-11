"""Unit tests for fail-closed Qdrant hybrid schema ownership."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models as qmodels


def test_ensure_hybrid_collection_creates_when_missing(monkeypatch):  # type: ignore[no-untyped-def]
    """Ensure ensure_hybrid_collection creates a new schema when absent."""
    smod = importlib.import_module("src.utils.storage")

    calls: list[dict[str, Any]] = []

    class _Client:
        """Stub client that triggers create_collection when missing."""

        def collection_exists(self, _name: str) -> bool:  # type: ignore[no-untyped-def]
            return False

        def create_collection(self, **kwargs: Any):  # type: ignore[no-untyped-def]
            calls.append(kwargs)

    result = smod.ensure_hybrid_collection(_Client(), "c", dense_dim=128)
    assert len(calls) == 1
    assert set(calls[0]["vectors_config"]) == {smod.DENSE_VECTOR_NAME}
    assert set(calls[0]["sparse_vectors_config"]) == {smod.SPARSE_VECTOR_NAME}
    assert result.compatible is True
    assert result.action == "created"


def test_ensure_hybrid_collection_leaves_compatible_schema_unchanged() -> None:
    """A compatible non-empty collection is accepted without mutation."""
    smod = importlib.import_module("src.utils.storage")

    class _Client:
        def collection_exists(self, _name: str) -> bool:  # type: ignore[no-untyped-def]
            return True

        def get_collection(self, _n: str):  # type: ignore[no-untyped-def]
            params = SimpleNamespace(
                vectors={
                    "text-dense": SimpleNamespace(
                        size=64,
                        distance=qmodels.Distance.COSINE,
                    )
                },
                sparse_vectors={
                    "text-sparse": SimpleNamespace(modifier=qmodels.Modifier.IDF)
                },
            )
            return SimpleNamespace(
                config=SimpleNamespace(params=params),
                points_count=4,
            )

    result = smod.ensure_hybrid_collection(_Client(), "c", dense_dim=64)

    assert result.compatible is True
    assert result.action == "unchanged"
    assert result.point_count == 4


def test_check_hybrid_collection_rejects_non_cosine_dense_vector() -> None:
    """A named dense vector with the wrong distance is incompatible."""
    smod = importlib.import_module("src.utils.storage")

    class _Client:
        def collection_exists(self, _name: str) -> bool:
            return True

        def get_collection(self, _name: str) -> SimpleNamespace:
            params = SimpleNamespace(
                vectors={
                    "text-dense": SimpleNamespace(
                        size=64,
                        distance=qmodels.Distance.DOT,
                    )
                },
                sparse_vectors={
                    "text-sparse": SimpleNamespace(modifier=qmodels.Modifier.IDF)
                },
            )
            return SimpleNamespace(
                config=SimpleNamespace(params=params),
                points_count=0,
            )

    result = smod.check_hybrid_collection(_Client(), "c", dense_dim=64)

    assert result.compatible is False
    assert result.action == "blocked"
    assert result.reason == "text_dense_distance_mismatch"


@pytest.mark.parametrize(
    ("distance", "compatible", "action", "reason"),
    [
        (qmodels.Distance.COSINE, True, "unchanged", "compatible"),
        (
            qmodels.Distance.DOT,
            False,
            "blocked",
            "text_dense_distance_mismatch",
        ),
    ],
)
def test_ensure_hybrid_collection_revalidates_after_create_error(
    distance: qmodels.Distance,
    compatible: bool,
    action: str,
    reason: str,
) -> None:
    """A concurrent creator wins only when its collection is canonical."""
    smod = importlib.import_module("src.utils.storage")

    class _Client:
        exists = False

        def collection_exists(self, _name: str) -> bool:
            return self.exists

        def create_collection(self, **_kwargs: Any) -> None:
            self.exists = True
            raise ValueError("collection already exists")

        def get_collection(self, _name: str) -> SimpleNamespace:
            params = SimpleNamespace(
                vectors={"text-dense": SimpleNamespace(size=64, distance=distance)},
                sparse_vectors={
                    "text-sparse": SimpleNamespace(modifier=qmodels.Modifier.IDF)
                },
            )
            return SimpleNamespace(
                config=SimpleNamespace(params=params),
                points_count=0,
            )

    result = smod.ensure_hybrid_collection(_Client(), "c", dense_dim=64)

    assert result.compatible is compatible
    assert result.action == action
    assert result.reason == reason


def test_ensure_hybrid_collection_blocks_nonempty_incompatible_schema() -> None:
    """A non-empty legacy collection is never mutated implicitly."""
    smod = importlib.import_module("src.utils.storage")

    class _Client:
        deleted = False
        created = False

        def collection_exists(self, _name: str) -> bool:
            return True

        def get_collection(self, _name: str) -> SimpleNamespace:
            params = SimpleNamespace(vectors=SimpleNamespace(size=64))
            return SimpleNamespace(
                config=SimpleNamespace(params=params),
                points_count=3,
            )

        def delete_collection(self, _name: str) -> None:
            self.deleted = True

        def create_collection(self, **_kwargs: Any) -> None:
            self.created = True

    client = _Client()
    result = smod.ensure_hybrid_collection(client, "legacy", dense_dim=64)

    assert result.compatible is False
    assert result.action == "blocked"
    assert result.reason == "legacy_unnamed_dense_vector"
    assert client.deleted is False
    assert client.created is False


def test_ensure_hybrid_collection_blocks_empty_incompatible_schema() -> None:
    """Runtime schema setup never deletes even an empty incompatible collection."""
    smod = importlib.import_module("src.utils.storage")
    calls = {"create": 0, "delete": 0}

    class _Client:
        def collection_exists(self, _name: str) -> bool:
            return True

        def get_collection(self, _name: str) -> SimpleNamespace:
            params = SimpleNamespace(vectors=SimpleNamespace(size=64))
            return SimpleNamespace(
                config=SimpleNamespace(params=params),
                points_count=0,
            )

        def delete_collection(self, _name: str) -> None:
            calls["delete"] += 1

        def create_collection(self, **_kwargs: Any) -> None:
            calls["create"] += 1

    result = smod.ensure_hybrid_collection(_Client(), "empty", dense_dim=64)

    assert result.compatible is False
    assert result.action == "blocked"
    assert calls == {"create": 0, "delete": 0}


def test_operator_rebuild_uses_exact_count_before_deletion() -> None:
    """The explicit operator path rebuilds only after an exact zero count."""
    smod = importlib.import_module("src.utils.storage")
    calls: dict[str, Any] = {"count": [], "create": 0, "delete": 0}

    class _Client:
        def collection_exists(self, _name: str) -> bool:
            return True

        def get_collection(self, _name: str) -> SimpleNamespace:
            params = SimpleNamespace(vectors=SimpleNamespace(size=64))
            return SimpleNamespace(
                config=SimpleNamespace(params=params),
                points_count=0,
            )

        def count(self, **kwargs: Any) -> SimpleNamespace:
            calls["count"].append(kwargs)
            return SimpleNamespace(count=0)

        def delete_collection(self, _name: str) -> None:
            calls["delete"] += 1

        def create_collection(self, **_kwargs: Any) -> None:
            calls["create"] += 1

    result = smod.rebuild_empty_hybrid_collection(_Client(), "empty", dense_dim=64)

    assert result.compatible is True
    assert result.action == "recreated"
    assert calls == {
        "count": [{"collection_name": "empty", "exact": True}],
        "create": 1,
        "delete": 1,
    }


def test_operator_rebuild_refuses_exact_nonempty_collection() -> None:
    """The explicit operator path never deletes a collection with exact points."""
    smod = importlib.import_module("src.utils.storage")

    class _Client:
        deleted = False

        def collection_exists(self, _name: str) -> bool:
            return True

        def get_collection(self, _name: str) -> SimpleNamespace:
            params = SimpleNamespace(vectors=SimpleNamespace(size=64))
            return SimpleNamespace(
                config=SimpleNamespace(params=params),
                points_count=0,
            )

        def count(self, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(count=2)

        def delete_collection(self, _name: str) -> None:
            self.deleted = True

    client = _Client()
    result = smod.rebuild_empty_hybrid_collection(client, "used", dense_dim=64)

    assert result.compatible is False
    assert result.action == "blocked"
    assert result.reason == "collection_nonempty_exact"
    assert result.point_count == 2
    assert client.deleted is False


def test_operator_rebuild_refuses_exact_count_error() -> None:
    """The explicit operator path fails closed when exact count fails."""
    smod = importlib.import_module("src.utils.storage")

    class _Client:
        deleted = False

        def collection_exists(self, _name: str) -> bool:
            return True

        def get_collection(self, _name: str) -> SimpleNamespace:
            params = SimpleNamespace(vectors=SimpleNamespace(size=64))
            return SimpleNamespace(
                config=SimpleNamespace(params=params),
                points_count=0,
            )

        def count(self, **_kwargs: Any) -> SimpleNamespace:
            raise ValueError("count unavailable")

        def delete_collection(self, _name: str) -> None:
            self.deleted = True

    client = _Client()
    result = smod.rebuild_empty_hybrid_collection(client, "unknown", dense_dim=64)

    assert result.compatible is False
    assert result.action == "error"
    assert result.reason == "exact_count_failed"
    assert client.deleted is False
