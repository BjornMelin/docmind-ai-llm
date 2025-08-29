"""Focused tests for SimpleCache edge cases to achieve maximum coverage.

These tests target specific uncovered lines in simple_cache.py to achieve
90%+ test coverage by testing error handling paths and edge conditions.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cache.simple_cache import SimpleCache


class TestCacheGetDocumentExceptionPaths:
    """Test exception handling in get_document method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_document_keyerror_handling(self):
        """Test KeyError handling in get_document (lines 65-68)."""
        cache = SimpleCache()

        # Mock the cache.get method to raise KeyError
        with patch.object(cache.cache, "get", side_effect=KeyError("Key not found")):
            result = await cache.get_document("test.pdf")

        assert result is None
        assert cache._misses == 1
        assert cache._hits == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_document_valueerror_handling(self):
        """Test ValueError handling in get_document (lines 65-68)."""
        cache = SimpleCache()

        # Mock the cache.get method to raise ValueError
        with patch.object(cache.cache, "get", side_effect=ValueError("Invalid value")):
            result = await cache.get_document("test.pdf")

        assert result is None
        assert cache._misses == 1
        assert cache._hits == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_document_oserror_handling(self):
        """Test OSError handling in get_document (lines 69-72)."""
        cache = SimpleCache()

        # Mock the _hash method to raise OSError
        with patch.object(cache, "_hash", side_effect=OSError("File system error")):
            result = await cache.get_document("test.pdf")

        assert result is None
        assert cache._misses == 1
        assert cache._hits == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_document_runtimeerror_handling(self):
        """Test RuntimeError handling in get_document (lines 69-72)."""
        cache = SimpleCache()

        # Mock the _hash method to raise RuntimeError
        with patch.object(cache, "_hash", side_effect=RuntimeError("Runtime error")):
            result = await cache.get_document("test.pdf")

        assert result is None
        assert cache._misses == 1
        assert cache._hits == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_document_attributeerror_handling(self):
        """Test AttributeError handling in get_document (lines 69-72)."""
        cache = SimpleCache()

        # Mock the cache.get method to raise AttributeError
        with patch.object(
            cache.cache, "get", side_effect=AttributeError("Attribute error")
        ):
            result = await cache.get_document("test.pdf")

        assert result is None
        assert cache._misses == 1
        assert cache._hits == 0


class TestCacheClearExceptionPaths:
    """Test exception handling in clear_cache method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_cache_file_deletion_branch(self, tmp_path):
        """Test clear_cache when database file exists (line 118->121)."""
        # Create cache with a real file
        cache_dir = tmp_path / "test_cache"
        cache = SimpleCache(cache_dir=str(cache_dir))

        # Store some data to create the file
        await cache.store_document("test.pdf", {"content": "test"})

        # Verify file exists
        db_path = Path(cache._persist_path)
        assert db_path.exists()

        # Clear cache
        result = await cache.clear_cache()

        assert result is True
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache._stored_documents == 0
        assert not db_path.exists()  # File should be deleted

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_cache_oserror_handling(self, tmp_path):
        """Test OSError handling in clear_cache (lines 126-128)."""
        cache_dir = tmp_path / "test_cache"
        cache = SimpleCache(cache_dir=str(cache_dir))

        # Store some data first to create the cache file
        await cache.store_document("test.pdf", {"content": "test"})

        # Mock Path.unlink to raise OSError
        with patch("pathlib.Path.unlink", side_effect=OSError("Permission denied")):
            result = await cache.clear_cache()

        assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_cache_valueerror_handling(self, tmp_path):
        """Test ValueError handling in clear_cache (lines 126-128)."""
        cache_dir = tmp_path / "test_cache"
        cache = SimpleCache(cache_dir=str(cache_dir))

        # Mock SimpleKVStore constructor to raise ValueError
        with patch(
            "src.cache.simple_cache.SimpleKVStore",
            side_effect=ValueError("Invalid cache"),
        ):
            result = await cache.clear_cache()

        assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_cache_runtimeerror_handling(self, tmp_path):
        """Test RuntimeError handling in clear_cache (lines 126-128)."""
        cache_dir = tmp_path / "test_cache"
        cache = SimpleCache(cache_dir=str(cache_dir))

        # Mock Path to raise RuntimeError during clear operation
        with patch("pathlib.Path", side_effect=RuntimeError("Runtime error")):
            result = await cache.clear_cache()

        assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_cache_attributeerror_handling(self, tmp_path):
        """Test AttributeError handling in clear_cache (lines 126-128)."""
        cache_dir = tmp_path / "test_cache"
        cache = SimpleCache(cache_dir=str(cache_dir))

        # Mock to raise AttributeError during clear operation
        with patch(
            "pathlib.Path.exists", side_effect=AttributeError("Attribute error")
        ):
            result = await cache.clear_cache()

        assert result is False


class TestCacheStatsExceptionPaths:
    """Test exception handling in get_cache_stats method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_cache_stats_exception_handling_simple(self):
        """Test basic exception handling in get_cache_stats (lines 149-151)."""
        cache = SimpleCache()

        # Patch the round function to raise an exception during stats calculation
        with patch("builtins.round", side_effect=ValueError("Round error")):
            stats = await cache.get_cache_stats()

        assert "error" in stats
        assert stats["cache_type"] == "simple_sqlite"
        assert "Round error" in stats["error"]


class TestCacheEdgeCaseScenarios:
    """Test additional edge cases for comprehensive coverage."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_with_corrupted_internal_state(self, tmp_path):
        """Test cache behavior when internal state is corrupted."""
        cache = SimpleCache(cache_dir=str(tmp_path))

        # Store a document first
        await cache.store_document("test.pdf", {"content": "test"})

        # Corrupt internal state by setting cache to None
        cache.cache = None

        # Should handle None cache gracefully
        result = await cache.get_document("test.pdf")
        assert result is None
        assert cache._misses == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_file_exists_but_unlink_fails(self, tmp_path):
        """Test specific scenario where file exists but unlink fails."""
        cache_dir = tmp_path / "test_cache"
        cache = SimpleCache(cache_dir=str(cache_dir))

        # Store data to create file
        await cache.store_document("test.pdf", {"content": "test"})

        # Verify file exists
        db_path = Path(cache._persist_path)
        assert db_path.exists()

        # Mock unlink to fail after exists check passes
        def failing_unlink(self):
            raise OSError("Permission denied")

        with patch.object(Path, "unlink", failing_unlink):
            # This should trigger the specific branch where exists() is True
            # but unlink() fails, covering line 118->121 and 126-128
            result = await cache.clear_cache()

        assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_division_by_zero_in_stats(self):
        """Test edge case of division by zero in hit rate calculation."""
        cache = SimpleCache()

        # Ensure no hits or misses have occurred
        assert cache._hits == 0
        assert cache._misses == 0

        # Get stats with zero total requests
        stats = await cache.get_cache_stats()

        # Should handle division by zero gracefully
        assert stats["hit_rate"] == 0.0
        assert stats["total_requests"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_kvstore_get_none_vs_exception(self):
        """Test distinction between None return and exception in cache.get()."""
        cache = SimpleCache()

        # Test case 1: cache.get returns None (normal miss)
        with patch.object(cache.cache, "get", return_value=None):
            result = await cache.get_document("test.pdf")
            assert result is None
            assert cache._misses == 1

        # Reset counters
        cache._misses = 0
        cache._hits = 0

        # Test case 2: cache.get raises KeyError (exception path)
        with patch.object(cache.cache, "get", side_effect=KeyError("Not found")):
            result = await cache.get_document("test.pdf")
            assert result is None
            assert cache._misses == 1


class TestCacheConcurrencyErrorPaths:
    """Test error handling in concurrent scenarios."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_filesystem_error_during_operations(self, tmp_path):
        """Test simple filesystem error handling."""
        cache_dir = tmp_path / "test_cache"
        cache = SimpleCache(cache_dir=str(cache_dir))

        # Test filesystem error during clear
        with patch("pathlib.Path.exists", side_effect=OSError("Filesystem error")):
            result = await cache.clear_cache()
            assert result is False  # Should handle filesystem error gracefully


class TestCacheHashingErrorScenarios:
    """Test error scenarios in cache key hashing."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_hash_with_inaccessible_file(self):
        """Test hashing when file exists but stat() fails."""
        cache = SimpleCache()

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Mock Path.stat to raise OSError
            with patch("pathlib.Path.stat", side_effect=OSError("Access denied")):
                # Should trigger the OSError path in get_document
                result = await cache.get_document(tmp_path)

            assert result is None
            assert cache._misses == 1

        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_hash_with_path_access_error(self):
        """Test hashing when Path() construction fails."""
        cache = SimpleCache()

        # Mock Path constructor to raise ValueError
        with patch("pathlib.Path", side_effect=ValueError("Invalid path")):
            result = await cache.get_document("test.pdf")

        assert result is None
        assert cache._misses == 1


class TestCacheInitializationErrorPaths:
    """Test error handling during cache initialization."""

    @pytest.mark.unit
    def test_cache_init_with_from_persist_path_exception(self, tmp_path):
        """Test cache initialization when from_persist_path fails."""
        cache_dir = tmp_path / "init_test"
        cache_dir.mkdir()
        db_path = cache_dir / "docmind.db"

        # Create a file with some content to trigger from_persist_path
        db_path.write_text('{"some": "data"}')

        # Mock from_persist_path to raise ValueError
        with patch(
            "llama_index.core.storage.kvstore.SimpleKVStore.from_persist_path",
            side_effect=ValueError("Persistence error"),
        ):
            # Should handle the error and create new cache
            cache = SimpleCache(cache_dir=str(cache_dir))

        # Should still be functional
        assert isinstance(cache, SimpleCache)
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache._stored_documents == 0
