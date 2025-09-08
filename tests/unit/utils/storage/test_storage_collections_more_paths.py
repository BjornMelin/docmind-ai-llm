"""Additional tests for storage collection utilities (edge/error paths)."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_get_collection_info_exists(monkeypatch):
    from src.utils import storage as s

    class _Info:
        def __init__(self):
            self.points_count = 10
            self.status = "green"
            self.config = SimpleNamespace(
                params=SimpleNamespace(
                    vectors={"text-dense": {}}, sparse_vectors={"text-sparse": {}}
                )
            )

    class _C:
        def collection_exists(self, name):
            return name == "ok"

        def get_collection(self, _):
            return _Info()

        def close(self):
            return None

    class _CM:
        def __enter__(self):
            return _C()

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(s, "create_sync_client", lambda: _CM())
    out = s.get_collection_info("ok")
    assert out["exists"] is True
    assert out["points_count"] == 10
    assert out["status"] == "green"


def test_get_collection_info_not_exists(monkeypatch):
    from src.utils import storage as s

    class _C:
        def collection_exists(self, _):
            return False

        def close(self):
            return None

    class _CM:
        def __enter__(self):
            return _C()

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(s, "create_sync_client", lambda: _CM())
    out = s.get_collection_info("missing")
    assert out == {"exists": False, "error": "Collection not found"}


def test_clear_collection_recreate(monkeypatch):
    from src.utils import storage as s

    class _Info:
        def __init__(self):
            self.config = SimpleNamespace(
                params=SimpleNamespace(vectors={}, sparse_vectors={})
            )

    class _C:
        def __init__(self):
            self.deleted = False
            self.created = False

        def collection_exists(self, _):
            return True

        def get_collection(self, _):
            return _Info()

        def delete_collection(self, _):
            self.deleted = True

        def create_collection(self, **_):
            self.created = True

        def close(self):
            return None

    class _CM:
        def __enter__(self):
            return _C()

        def __exit__(self, *_):
            return False

    cm = _CM()
    monkeypatch.setattr(s, "create_sync_client", lambda: cm)
    assert s.clear_collection("col") is True
