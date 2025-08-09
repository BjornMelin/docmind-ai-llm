"""Unit tests for critical resource management fixes (P0).

Tests to verify that the memory leaks and resource management issues
identified in the code review have been properly fixed.

Test Coverage:
    - Async context managers for Qdrant clients
    - GPU memory cleanup after operations
    - File handle cleanup in document loading
    - Proper resource cleanup in exception scenarios
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch


@pytest.mark.asyncio
async def test_managed_async_qdrant_client_cleanup():
    """Test that managed_async_qdrant_client properly cleans up resources."""
    from utils.utils import managed_async_qdrant_client

    with patch("utils.utils.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Test normal operation
        async with managed_async_qdrant_client("http://localhost:6333") as client:
            assert client == mock_client

        # Verify client was closed
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_managed_async_qdrant_client_cleanup_on_exception():
    """Test that client is cleaned up even when exceptions occur."""
    from utils.utils import managed_async_qdrant_client

    with patch("utils.utils.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Test exception handling
        with pytest.raises(ValueError):
            async with managed_async_qdrant_client("http://localhost:6333") as client:
                raise ValueError("Test exception")

        # Verify client was still closed despite exception
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_managed_gpu_operation_cleanup():
    """Test that GPU operations are properly cleaned up."""
    from utils.utils import managed_gpu_operation

    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.synchronize") as mock_sync:
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                with patch("gc.collect") as mock_gc:
                    # Test normal operation
                    async with managed_gpu_operation():
                        pass  # Simulate GPU operation

                    # Verify cleanup was called
                    mock_sync.assert_called_once()
                    mock_empty_cache.assert_called_once()
                    mock_gc.assert_called_once()


@pytest.mark.asyncio
async def test_managed_gpu_operation_cleanup_on_exception():
    """Test that GPU cleanup occurs even when operations fail."""
    from utils.utils import managed_gpu_operation

    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.synchronize") as mock_sync:
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                with patch("gc.collect") as mock_gc:
                    # Test exception handling
                    with pytest.raises(RuntimeError):
                        async with managed_gpu_operation():
                            raise RuntimeError("GPU operation failed")

                    # Verify cleanup still occurred
                    mock_sync.assert_called_once()
                    mock_empty_cache.assert_called_once()
                    mock_gc.assert_called_once()


@pytest.mark.asyncio
async def test_create_index_async_resource_management():
    """Test that create_index_async uses proper resource management."""
    from llama_index.core import Document

    from utils.index_builder import create_index_async

    with patch("utils.index_builder.managed_async_qdrant_client") as mock_context:
        with patch("utils.index_builder.setup_hybrid_qdrant_async") as mock_setup:
            with patch(
                "utils.index_builder.VectorStoreIndex.from_documents"
            ) as mock_index:
                with patch("utils.index_builder.ensure_spacy_model"):
                    # Mock the async context manager
                    mock_client = AsyncMock()
                    mock_context.return_value.__aenter__.return_value = mock_client
                    mock_context.return_value.__aexit__.return_value = None

                    # Mock other dependencies
                    mock_setup.return_value = MagicMock()
                    mock_index.return_value = MagicMock()

                    # Test the function
                    docs = [Document(text="test document")]
                    result = await create_index_async(docs, use_gpu=False)

                    # Verify context manager was used
                    mock_context.assert_called_once()
                    mock_context.return_value.__aenter__.assert_called_once()
                    mock_context.return_value.__aexit__.assert_called_once()

                    # Verify result structure
                    assert "vector" in result
                    assert "kg" in result
                    assert "retriever" in result


def test_extract_images_from_pdf_resource_management():
    """Test that PDF image extraction properly manages file resources."""
    from utils.document_loader import extract_images_from_pdf

    # Create a temporary file to simulate PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        with patch("fitz.open") as mock_fitz_open:
            # Mock fitz document with context manager support
            mock_doc = MagicMock()
            mock_doc.__enter__ = MagicMock(return_value=mock_doc)
            mock_doc.__exit__ = MagicMock(return_value=None)
            mock_doc.__len__ = MagicMock(return_value=0)  # No pages
            mock_fitz_open.return_value = mock_doc

            # Test the function
            result = extract_images_from_pdf(tmp_path)

            # Verify context manager was used
            mock_fitz_open.assert_called_once_with(tmp_path)
            mock_doc.__enter__.assert_called_once()
            mock_doc.__exit__.assert_called_once()

            # Verify result is a list
            assert isinstance(result, list)

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_extract_images_from_pdf_file_not_found():
    """Test that extract_images_from_pdf handles missing files gracefully."""
    from utils.document_loader import extract_images_from_pdf

    # Test with non-existent file
    result = extract_images_from_pdf("/nonexistent/file.pdf")

    # Should return empty list, not crash
    assert result == []


@pytest.mark.asyncio
async def test_connection_pool_resource_management():
    """Test that QdrantConnectionPool properly manages connections."""
    from utils.utils import QdrantConnectionPool

    # Reset singleton instance for test
    QdrantConnectionPool._instance = None

    with patch("utils.utils.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        pool = QdrantConnectionPool()
        pool.configure("http://localhost:6333", max_size=2)

        # Get client from pool
        client = await pool.get_client()
        assert client == mock_client

        # Return client to pool
        await pool.return_client(client)

        # Verify client is in the pool (not closed)
        assert len(pool.clients) == 1

        # Get client again (should reuse from pool)
        client2 = await pool.get_client()
        assert client2 == mock_client

        # Verify pool is empty again
        assert len(pool.clients) == 0

        # Close all connections (should close the remaining clients in pool if any)
        await pool.return_client(client2)  # Put it back first
        await pool.close_all()

        # Now it should be called once when close_all() runs
        mock_client.close.assert_called_once()


def test_load_documents_llama_temp_file_cleanup():
    """Test that temporary files are cleaned up in load_documents_llama."""
    from utils.document_loader import load_documents_llama

    # Mock file object
    mock_file = MagicMock()
    mock_file.name = "test.txt"
    mock_file.getvalue.return_value = b"test content"

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        with patch("utils.document_loader.SimpleDirectoryReader") as mock_reader:
            with patch("utils.document_loader.LlamaParse") as mock_llama_parse:
                with patch("os.remove") as mock_remove:
                    # Mock temporary file context manager
                    mock_temp_file = MagicMock()
                    mock_temp_file.name = "/tmp/test_temp_file.txt"
                    mock_temp.__enter__ = MagicMock(return_value=mock_temp_file)
                    mock_temp.__exit__ = MagicMock(return_value=None)
                    mock_temp.return_value = mock_temp

                    # Mock LlamaParse parser
                    mock_parser = MagicMock()
                    mock_llama_parse.return_value = mock_parser

                    # Mock document reader
                    mock_reader.return_value.load_data.return_value = [
                        MagicMock(text="test", metadata={})
                    ]

                    # Test the function
                    result = load_documents_llama([mock_file])

                    # Verify temporary file cleanup was called
                    mock_remove.assert_called()
                    assert len(result) > 0


def test_load_documents_llama_video_resource_cleanup():
    """Test that video resources are cleaned up properly."""
    from utils.document_loader import load_documents_llama

    # Mock video file
    mock_file = MagicMock()
    mock_file.name = "test.mp4"
    mock_file.type = "video/mp4"
    mock_file.getvalue.return_value = b"video content"

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        with patch("moviepy.video.io.VideoFileClip.VideoFileClip") as mock_video:
            with patch("utils.document_loader.whisper_load") as mock_whisper:
                with patch("utils.document_loader.LlamaParse") as mock_llama_parse:
                    with patch("os.remove") as mock_remove:
                        with patch("os.path.exists", return_value=True):
                            # Mock LlamaParse parser
                            mock_parser = MagicMock()
                            mock_llama_parse.return_value = mock_parser

                            # Mock temporary file
                            mock_temp_file = MagicMock()
                            mock_temp_file.name = "/tmp/test_video.mp4"
                            mock_temp.__enter__ = MagicMock(return_value=mock_temp_file)
                            mock_temp.__exit__ = MagicMock(return_value=None)
                            mock_temp.return_value = mock_temp

                            # Mock video clip
                            mock_clip = MagicMock()
                            mock_clip.duration = 10
                            mock_clip.audio.write_audiofile = MagicMock()
                            mock_clip.close = MagicMock()
                            mock_video.return_value = mock_clip

                            # Mock whisper
                            mock_model = MagicMock()
                            mock_model.transcribe.return_value = {
                                "text": "transcribed text"
                            }
                            mock_whisper.return_value = mock_model

                            # Test the function
                            result = load_documents_llama([mock_file], parse_media=True)

                            # Verify video clip was closed
                            mock_clip.close.assert_called_once()

                            # Verify temp files were removed
                            assert (
                                mock_remove.call_count >= 1
                            )  # At least main file cleanup


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_memory_cleanup_integration():
    """Integration test for GPU memory cleanup (requires CUDA)."""
    import torch

    # Get initial GPU memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()

        # Create some tensors to use GPU memory
        large_tensor = torch.randn(1000, 1000, device="cuda")
        after_allocation = torch.cuda.memory_allocated()

        # Verify memory increased
        assert after_allocation > initial_memory

        # Cleanup
        del large_tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        final_memory = torch.cuda.memory_allocated()

        # Memory should be cleaned up (may not be exactly equal due to fragmentation)
        assert final_memory <= after_allocation


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
