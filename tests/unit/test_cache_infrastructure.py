"""Enhanced tests for SimpleCache infrastructure - persistence, corruption, and edge cases.

Focuses on areas not fully covered by existing tests:
- Disk persistence failure scenarios
- Cache corruption recovery
- Path validation edge cases
- Memory pressure scenarios
- Thread safety with file locks
"""

import stat
import threading
import time

import pytest
from llama_index.core.storage.kvstore import SimpleKVStore

from src.cache.simple_cache import SimpleCache


class TestCachePersistenceFailures:
    """Test cache persistence failure scenarios."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_readonly_cache_directory(self, tmp_path):
        """Test handling when cache directory becomes read-only."""
        cache_dir = tmp_path / "readonly_cache"
        cache_dir.mkdir()

        # Initialize cache first
        cache = SimpleCache(cache_dir=str(cache_dir))

        # Store initial document
        await cache.store_document("test.pdf", {"content": "test"})

        # Make directory read-only
        cache_dir.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        try:
            # Should handle read-only gracefully
            result = await cache.store_document("readonly.pdf", {"content": "readonly"})
            # On Linux, this might fail, on Windows it might succeed
            assert isinstance(result, bool)

            # Reading should still work
            existing = await cache.get_document("test.pdf")
            assert existing is not None
        finally:
            # Restore write permissions for cleanup
            cache_dir.chmod(stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_disk_full_simulation(self, tmp_path, mocker):
        """Test handling disk full scenarios."""
        cache = SimpleCache(cache_dir=str(tmp_path))

        # Mock persist to raise OSError (disk full)
        mock_persist = mocker.patch.object(
            SimpleKVStore, "persist", side_effect=OSError("No space left on device")
        )

        # Should handle disk full gracefully
        result = await cache.store_document("diskfull.pdf", {"content": "large data"})
        assert result is False

        mock_persist.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_corrupted_cache_file_recovery(self, tmp_path):
        """Test recovery from corrupted cache files."""
        cache_dir = tmp_path / "corrupted_cache"
        cache_dir.mkdir()
        db_path = cache_dir / "docmind.db"

        # Create corrupted cache file
        with open(db_path, "w") as f:
            f.write("corrupted json data {invalid")

        # Should create new cache when encountering corruption
        cache = SimpleCache(cache_dir=str(cache_dir))

        # Should be able to store/retrieve after recovery
        await cache.store_document("recovery.pdf", {"content": "recovered"})
        result = await cache.get_document("recovery.pdf")

        assert result is not None
        assert result["content"] == "recovered"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_file_access_with_locks(self, tmp_path):
        """Test concurrent file access with potential lock scenarios."""
        shared_cache_dir = str(tmp_path / "shared")

        results = []
        errors = []

        async def store_operation(cache_id: int):
            """Store operations from different cache instances."""
            try:
                cache = SimpleCache(cache_dir=shared_cache_dir)
                success = await cache.store_document(
                    f"concurrent_{cache_id}.pdf",
                    {"id": cache_id, "data": f"data_{cache_id}"},
                )
                results.append((cache_id, success))
            except Exception as e:
                errors.append((cache_id, str(e)))

        # Run 5 concurrent cache instances
        import asyncio

        await asyncio.gather(*[store_operation(i) for i in range(5)])

        # All operations should complete (some may fail due to concurrency)
        assert len(results) + len(errors) == 5
        # At least some should succeed
        successful = [r for r in results if r[1] is True]
        assert len(successful) > 0


class TestCachePathValidation:
    """Test path validation and sanitization edge cases."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unicode_paths(self):
        """Test handling of unicode characters in paths."""
        cache = SimpleCache()

        unicode_paths = [
            "test_文档.pdf",  # Chinese
            "тест.pdf",  # Cyrillic
            "тест_файл.pdf",  # Mixed
            "émoji_😀.pdf",  # Emoji
        ]

        for path in unicode_paths:
            # Should handle unicode paths without error
            await cache.store_document(path, {"content": f"unicode test for {path}"})
            result = await cache.get_document(path)
            assert result is not None
            assert result["content"] == f"unicode test for {path}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extremely_long_paths(self):
        """Test handling of extremely long file paths."""
        cache = SimpleCache()

        # Create path at OS limit (typically 255 for filename, 4096 for full path)
        long_filename = "x" * 250 + ".pdf"
        very_long_path = "/very/long/path/" + "subdir/" * 50 + long_filename

        # Should raise ValueError for paths that are too long
        with pytest.raises(ValueError, match="path is too long"):
            await cache.store_document(very_long_path, {"content": "test"})

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_special_character_paths(self):
        """Test handling of special characters in paths."""
        cache = SimpleCache()

        special_paths = [
            "test with spaces.pdf",
            "test-with-dashes.pdf",
            "test_with_underscores.pdf",
            "test.multiple.dots.pdf",
            "test(with)parens.pdf",
            "test[with]brackets.pdf",
        ]

        for path in special_paths:
            await cache.store_document(
                path, {"content": f"special chars test for {path}"}
            )
            result = await cache.get_document(path)
            assert result is not None


class TestCacheMemoryPressure:
    """Test cache behavior under memory pressure scenarios."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_large_document_storage(self):
        """Test storing very large documents."""
        cache = SimpleCache()

        # Create 5MB document content
        large_content = "x" * (5 * 1024 * 1024)
        large_doc = {
            "content": large_content,
            "metadata": {"size": len(large_content), "type": "large_test"},
        }

        # Should handle large documents
        result = await cache.store_document("large_doc.pdf", large_doc)
        assert result is True

        # Should retrieve successfully
        retrieved = await cache.get_document("large_doc.pdf")
        assert retrieved is not None
        assert len(retrieved["content"]) == 5 * 1024 * 1024

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_many_small_documents(self):
        """Test storing many small documents."""
        cache = SimpleCache()

        # Store 1000 small documents
        document_count = 1000
        for i in range(document_count):
            doc_data = {
                "id": i,
                "content": f"Small document {i} content",
                "metadata": {"index": i, "batch": "small_docs"},
            }
            await cache.store_document(f"small_doc_{i}.pdf", doc_data)

        # Verify random samples
        sample_indices = [0, 100, 500, 999]
        for i in sample_indices:
            result = await cache.get_document(f"small_doc_{i}.pdf")
            assert result is not None
            assert result["id"] == i
            assert f"Small document {i}" in result["content"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_nested_data_structures(self):
        """Test handling of deeply nested data structures."""
        cache = SimpleCache()

        # Create deeply nested structure
        nested_data = {"level_0": {}}
        current_level = nested_data["level_0"]

        for i in range(20):  # 20 levels deep
            current_level[f"level_{i + 1}"] = {
                "data": f"data_at_level_{i + 1}",
                "items": [f"item_{j}" for j in range(5)],
                "nested": {},
            }
            current_level = current_level[f"level_{i + 1}"]["nested"]

        # Should handle deeply nested structures
        await cache.store_document("nested_doc.pdf", nested_data)
        result = await cache.get_document("nested_doc.pdf")

        assert result is not None
        # Verify structure integrity
        assert "level_0" in result
        assert "level_1" in result["level_0"]


class TestCacheThreadSafety:
    """Test cache thread safety and race condition handling."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_read_write_same_document(self):
        """Test concurrent read/write operations on same document."""
        cache = SimpleCache()
        doc_path = "concurrent_test.pdf"

        # Pre-populate document
        await cache.store_document(doc_path, {"version": 0, "content": "initial"})

        write_results = []
        read_results = []

        async def write_operation(version: int):
            """Write with version number."""
            result = await cache.store_document(
                doc_path, {"version": version, "content": f"version_{version}"}
            )
            write_results.append((version, result))

        async def read_operation(read_id: int):
            """Read operation."""
            result = await cache.get_document(doc_path)
            read_results.append((read_id, result))

        # Mix of concurrent reads and writes
        import asyncio

        operations = []

        # 3 concurrent writes
        for i in range(1, 4):
            operations.append(write_operation(i))

        # 5 concurrent reads
        for i in range(5):
            operations.append(read_operation(i))

        await asyncio.gather(*operations, return_exceptions=True)

        # All operations should complete
        assert len(write_results) == 3
        assert len(read_results) == 5

        # Final state should be consistent
        final_result = await cache.get_document(doc_path)
        assert final_result is not None
        assert "version" in final_result

    @pytest.mark.unit
    def test_thread_safety_with_threading_module(self):
        """Test thread safety using threading module."""
        cache = SimpleCache()
        results = []
        errors = []

        def thread_operation(thread_id: int):
            """Synchronous thread operation."""
            try:
                import asyncio

                # Create new event loop for thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def async_work():
                    await cache.store_document(
                        f"thread_{thread_id}.pdf",
                        {"thread_id": thread_id, "content": f"thread_{thread_id}_data"},
                    )
                    result = await cache.get_document(f"thread_{thread_id}.pdf")
                    return result

                result = loop.run_until_complete(async_work())
                results.append((thread_id, result))
                loop.close()

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start 5 threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=thread_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout

        # All threads should complete successfully
        assert len(results) == 5
        assert len(errors) == 0

        # Verify data integrity
        for thread_id, result in results:
            assert result is not None
            assert result["thread_id"] == thread_id


class TestCacheStatsAccuracy:
    """Test cache statistics accuracy under various conditions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_hit_miss_ratio_accuracy(self, tmp_path):
        """Test hit/miss ratio calculation accuracy."""
        cache = SimpleCache(cache_dir=str(tmp_path))

        # Store 5 documents
        for i in range(5):
            await cache.store_document(f"doc_{i}.pdf", {"id": i})

        # 10 hits, 10 misses
        for i in range(10):
            # Hit - existing document
            await cache.get_document(f"doc_{i % 5}.pdf")

            # Miss - non-existent document
            await cache.get_document(f"missing_{i}.pdf")

        stats = await cache.get_cache_stats()

        # Should have 50% hit rate (10 hits, 10 misses)
        assert stats["hits"] == 10
        assert stats["misses"] == 10
        assert stats["total_requests"] == 20
        assert stats["hit_rate"] == 0.5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_document_count_accuracy_after_updates(self, tmp_path):
        """Test document count accuracy when updating existing documents."""
        cache = SimpleCache(cache_dir=str(tmp_path))

        # Store 3 new documents
        await cache.store_document("doc1.pdf", {"version": 1})
        await cache.store_document("doc2.pdf", {"version": 1})
        await cache.store_document("doc3.pdf", {"version": 1})

        stats = await cache.get_cache_stats()
        assert stats["total_documents"] == 3

        # Update existing documents (should not increase count)
        await cache.store_document("doc1.pdf", {"version": 2})
        await cache.store_document("doc2.pdf", {"version": 2})

        stats = await cache.get_cache_stats()
        assert stats["total_documents"] == 3  # Should still be 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stats_after_cache_clear(self, tmp_path):
        """Test stats reset after cache clear."""
        cache = SimpleCache(cache_dir=str(tmp_path))

        # Build up cache statistics
        for i in range(5):
            await cache.store_document(f"doc_{i}.pdf", {"id": i})
            await cache.get_document(f"doc_{i}.pdf")  # Hit
            await cache.get_document(f"missing_{i}.pdf")  # Miss

        # Verify stats before clear
        stats = await cache.get_cache_stats()
        assert stats["total_documents"] == 5
        assert stats["hits"] == 5
        assert stats["misses"] == 5

        # Clear cache
        await cache.clear_cache()

        # Stats should be reset
        stats = await cache.get_cache_stats()
        assert stats["total_documents"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0


class TestCacheHashingEdgeCases:
    """Test cache key hashing edge cases."""

    @pytest.mark.unit
    def test_hash_consistency_with_file_changes(self, tmp_path):
        """Test hash changes when file attributes change."""
        cache = SimpleCache()

        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_text("original content")
        original_hash = cache._hash(str(test_file))

        # Modify file content (changes size and mtime)
        time.sleep(0.01)  # Ensure mtime difference
        test_file.write_text("modified content with different size")
        modified_hash = cache._hash(str(test_file))

        # Hash should change
        assert original_hash != modified_hash

    @pytest.mark.unit
    def test_hash_for_nonexistent_files(self):
        """Test hashing behavior for nonexistent files."""
        cache = SimpleCache()

        # Nonexistent files should have consistent hashes based on name
        hash1 = cache._hash("/nonexistent/file1.pdf")
        hash2 = cache._hash("/nonexistent/file1.pdf")
        hash3 = cache._hash("/nonexistent/file2.pdf")

        # Same path should give same hash
        assert hash1 == hash2

        # Different paths should give different hashes
        assert hash1 != hash3

    @pytest.mark.unit
    def test_hash_collision_resistance(self):
        """Test hash collision resistance with similar paths."""
        cache = SimpleCache()

        similar_paths = [
            "/path/to/document.pdf",
            "/path/to/document_.pdf",
            "/path/to/document1.pdf",
            "/path/to/Document.pdf",
            "/path/to/DOCUMENT.pdf",
        ]

        hashes = [cache._hash(path) for path in similar_paths]

        # All hashes should be unique
        assert len(set(hashes)) == len(similar_paths)
