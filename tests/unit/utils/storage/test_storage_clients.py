"""Unit tests for storage client creation helpers.

Covers create_sync_client and create_async_client happy paths.
"""

import pytest

pytestmark = pytest.mark.unit


def test_create_sync_client(monkeypatch):
    from src.utils import storage as storage_mod

    captured = {}

    class _C:
        def __init__(self, **kw):
            captured.update(kw)

        def close(self):
            return None

    monkeypatch.setattr(storage_mod, "QdrantClient", _C)

    with storage_mod.create_sync_client() as client:
        assert client is not None
    # Config propagated
    assert "url" in captured
    assert "timeout" in captured


@pytest.mark.asyncio
async def test_create_async_client(monkeypatch):
    from src.utils import storage as storage_mod

    captured = {}

    class _A:
        def __init__(self, **kw):
            captured.update(kw)

        async def close(self):
            return None

    monkeypatch.setattr(storage_mod, "AsyncQdrantClient", _A)

    async with storage_mod.create_async_client() as client:
        assert client is not None
    assert "url" in captured
    assert "timeout" in captured
