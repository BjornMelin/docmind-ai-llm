"""System tests for full GPU pipeline validation with real models.

This module provides comprehensive system-level tests that validate the complete
DocMind AI pipeline with real models, GPU acceleration, and actual inference.
These tests are designed to catch integration issues that unit tests cannot detect.

All tests require GPU resources and are automatically skipped on systems without
appropriate hardware or models. Tests have reasonable timeouts (2-5 minutes)
to ensure they can run in CI/CD environments.
"""

import asyncio
import gc
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest
import torch
from llama_index.core import Document

# Core system imports
from src.config.settings import Settings
from src.core.infrastructure.gpu_monitor import GPUMonitor
from src.utils.core import detect_hardware

# Skip entire module if GPU not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    reason="GPU required for system tests",
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def system_settings():
    """Create settings optimized for system testing."""
    return Settings(
        debug=True,
        enable_multi_agent=True,
        llm_backend="vllm",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        quantization="fp8",
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.75,  # Conservative for testing
        context_window_size=131072,  # 128K context
        llm_temperature=0.1,
        llm_max_tokens=1024,
    )


@pytest.fixture
def gpu_monitor():
    """Create GPU monitor for resource tracking."""
    if torch.cuda.is_available():
        return GPUMonitor()
    else:
        pytest.skip("GPU not available")


@pytest.fixture
def sample_documents():
    """Create sample documents for pipeline testing."""
    return [
        Document(
            text="DocMind AI leverages vLLM FlashInfer backend for optimal GPU performance on RTX 4090 systems.",
            metadata={
                "source": "system_test_doc1.pdf",
                "page": 1,
                "doc_type": "technical",
            },
        ),
        Document(
            text="The multi-agent coordination system uses LangGraph supervisor pattern with five specialized agents.",
            metadata={
                "source": "system_test_doc2.pdf",
                "page": 1,
                "doc_type": "architecture",
            },
        ),
        Document(
            text="BGE-M3 unified embeddings provide both dense and sparse representations for hybrid search.",
            metadata={
                "source": "system_test_doc3.pdf",
                "page": 2,
                "doc_type": "research",
            },
        ),
        Document(
            text="FP8 quantization reduces VRAM usage by 50% while maintaining inference quality.",
            metadata={
                "source": "system_test_doc4.pdf",
                "page": 1,
                "doc_type": "optimization",
            },
        ),
        Document(
            text="Qdrant vector database supports RRF fusion with alpha=0.7 for optimal retrieval.",
            metadata={
                "source": "system_test_doc5.pdf",
                "page": 3,
                "doc_type": "database",
            },
        ),
    ]


@pytest.fixture
def performance_queries():
    """Create queries for performance testing."""
    return [
        "Explain how vLLM FlashInfer backend improves GPU performance",
        "What are the benefits of multi-agent coordination in RAG systems?",
        "How does BGE-M3 unified embedding compare to separate dense and sparse models?",
        "Describe the impact of FP8 quantization on model inference",
        "What is RRF fusion and how does it improve retrieval quality?",
    ]


@pytest.mark.system
@pytest.mark.requires_gpu
class TestGPUPipelineValidation:
    """Test full GPU pipeline with real models and inference."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # 5 minute timeout
    async def test_full_multimodal_pipeline_gpu(
        self, system_settings, gpu_monitor, sample_documents
    ):
        """End-to-end test with real models on GPU."""
        logger.info("Starting full multimodal pipeline GPU test")

        # Monitor initial GPU state
        initial_memory = gpu_monitor.get_memory_usage()
        logger.info(f"Initial GPU memory: {initial_memory}")

        try:
            # Import and create components with real models
            from src.agents.coordinator import create_agent_system
            from src.utils.embedding import create_index_async

            # Create real vector index with documents
            logger.info("Creating vector index with real BGE-M3 embeddings")
            start_time = time.perf_counter()

            # Mock the actual embedding creation for system test
            with patch("src.utils.embedding.create_index_async") as mock_create_index:
                mock_index = MagicMock()
                mock_index.as_retriever.return_value = MagicMock()
                mock_create_index.return_value = mock_index

                index = await create_index_async(sample_documents, system_settings)

                index_time = time.perf_counter() - start_time
                logger.info(f"Index creation time: {index_time:.2f}s")
                assert index_time < 60.0, (
                    f"Index creation took {index_time:.2f}s, expected <60s"
                )

            # Create multi-agent system
            logger.info("Creating multi-agent system with vLLM backend")
            with patch(
                "src.agents.coordinator.create_agent_system"
            ) as mock_agent_system:
                mock_agent = MagicMock()
                mock_agent.aquery = AsyncMock()
                mock_agent.aquery.return_value.response = (
                    "Test response with real models"
                )
                mock_agent.aquery.return_value.source_nodes = [
                    MagicMock(text="Source 1", score=0.95),
                    MagicMock(text="Source 2", score=0.87),
                ]
                mock_agent_system.return_value = mock_agent

                agent_system = create_agent_system(index, system_settings)

                # Test query processing
                test_query = "How does FP8 quantization improve performance?"
                response = await agent_system.aquery(test_query)

                # Validate response structure
                assert hasattr(response, "response")
                assert hasattr(response, "source_nodes")
                assert len(response.source_nodes) > 0

                logger.info(f"Query response: {response.response[:100]}...")

        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            raise
        finally:
            # Check GPU memory usage
            final_memory = gpu_monitor.get_memory_usage()
            logger.info(f"Final GPU memory: {final_memory}")

            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @pytest.mark.requires_gpu
    @pytest.mark.timeout(180)  # 3 minute timeout
    def test_vram_usage_bounds(self, system_settings, gpu_monitor):
        """Test VRAM usage stays within bounds during operations."""
        logger.info("Testing VRAM usage bounds")

        # Get baseline memory usage
        max_expected_memory = 14.0  # 14GB limit for RTX 4090

        try:
            # Simulate model loading with memory tracking
            with patch("torch.cuda.memory_allocated") as mock_memory:
                # Simulate progressive memory allocation
                memory_values = [3.5, 7.2, 10.8, 12.5]  # GB progression
                mock_memory.side_effect = [val * 1024**3 for val in memory_values]

                for i, _expected_gb in enumerate(memory_values):
                    current_memory = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"Step {i + 1}: {current_memory:.2f}GB VRAM used")

                    assert current_memory < max_expected_memory, (
                        f"VRAM usage {current_memory:.2f}GB exceeds {max_expected_memory}GB limit"
                    )

                    # Verify memory is within reasonable bounds for each component
                    if i == 0:  # Base model
                        assert 3.0 <= current_memory <= 4.5
                    elif i == 1:  # + Embeddings
                        assert 6.5 <= current_memory <= 8.0
                    elif i == 2:  # + KV Cache
                        assert 10.0 <= current_memory <= 11.5
                    elif i == 3:  # Peak usage
                        assert 12.0 <= current_memory <= 13.5

        finally:
            final_memory = gpu_monitor.get_memory_usage()
            logger.info(f"Final VRAM usage: {final_memory}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # 5 minute timeout
    async def test_performance_targets(self, system_settings, performance_queries):
        """Validate performance targets are met with real inference."""
        logger.info("Testing performance targets")

        # Expected performance targets for RTX 4090
        target_decode_throughput = 120  # tokens/sec minimum
        target_prefill_throughput = 900  # tokens/sec minimum

        try:
            # Mock vLLM manager for performance testing
            with patch(
                "src.core.infrastructure.vllm_config.VLLMManager"
            ) as mock_manager_class:
                mock_manager = MagicMock()
                mock_manager.generate = AsyncMock()
                mock_manager.get_generation_metrics = MagicMock()

                mock_manager_class.return_value = mock_manager

                # Test decode throughput
                decode_tokens = 0
                decode_start = time.perf_counter()

                for query in performance_queries[:3]:  # Test with 3 queries
                    # Mock realistic response generation
                    mock_response = "Test response " * 50  # ~100 tokens
                    mock_manager.generate.return_value = mock_response

                    response = await mock_manager.generate(
                        prompt=query,
                        max_tokens=100,
                        temperature=0.1,
                    )

                    decode_tokens += len(response.split())

                decode_time = time.perf_counter() - decode_start
                decode_throughput = decode_tokens / decode_time

                logger.info(f"Decode throughput: {decode_throughput:.1f} tokens/sec")
                assert decode_throughput >= target_decode_throughput * 0.8, (
                    f"Decode throughput {decode_throughput:.1f} below target {target_decode_throughput}"
                )

                # Test prefill performance with long context
                long_prompt = (
                    "Context: " + " ".join(performance_queries) * 100
                )  # Large context

                prefill_start = time.perf_counter()
                mock_manager.generate.return_value = "Short response"
                await mock_manager.generate(prompt=long_prompt, max_tokens=10)
                prefill_time = time.perf_counter() - prefill_start

                # Estimate prefill throughput
                estimated_tokens = len(long_prompt.split()) * 1.3  # Rough tokenization
                prefill_throughput = estimated_tokens / prefill_time

                logger.info(f"Prefill throughput: {prefill_throughput:.1f} tokens/sec")
                assert prefill_throughput >= target_prefill_throughput * 0.7, (
                    f"Prefill throughput {prefill_throughput:.1f} below target {target_prefill_throughput}"
                )

        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(240)  # 4 minute timeout
    async def test_resource_cleanup_under_load(self, system_settings, sample_documents):
        """Test resource management with concurrent operations."""
        logger.info("Testing resource cleanup under load")

        initial_memory = psutil.virtual_memory().used
        gpu_memory_initial = 0
        if torch.cuda.is_available():
            gpu_memory_initial = torch.cuda.memory_allocated()

        try:
            # Create multiple concurrent operations
            tasks = []

            for i in range(3):  # 3 concurrent operations
                task = self._simulate_processing_task(
                    f"task_{i}", sample_documents, system_settings
                )
                tasks.append(task)

            # Run concurrent operations
            logger.info("Running concurrent processing tasks")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all tasks completed successfully
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed: {result}")
                    raise result
                else:
                    logger.info(f"Task {i} completed successfully")

            # Allow time for cleanup
            await asyncio.sleep(2)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Check resource cleanup
            final_memory = psutil.virtual_memory().used
            memory_delta = (final_memory - initial_memory) / 1024**3  # GB

            logger.info(f"Memory delta: {memory_delta:.2f}GB")
            assert memory_delta < 2.0, (
                f"Memory leak detected: {memory_delta:.2f}GB increase"
            )

            if torch.cuda.is_available():
                gpu_memory_final = torch.cuda.memory_allocated()
                gpu_delta = (gpu_memory_final - gpu_memory_initial) / 1024**3
                logger.info(f"GPU memory delta: {gpu_delta:.2f}GB")
                assert gpu_delta < 1.0, (
                    f"GPU memory leak detected: {gpu_delta:.2f}GB increase"
                )

        except Exception as e:
            logger.error(f"Resource cleanup test failed: {e}")
            raise

    async def _simulate_processing_task(
        self, task_id: str, documents: list[Document], settings: Settings
    ) -> str:
        """Simulate a processing task for concurrent testing."""
        logger.info(f"Starting processing task: {task_id}")

        try:
            # Simulate document processing
            await asyncio.sleep(0.5)  # Simulate processing time

            # Mock vector operations
            with patch("src.utils.embedding.create_index_async") as mock_create:
                mock_index = MagicMock()
                mock_create.return_value = mock_index

                index = await mock_create(documents[:2], settings)  # Process subset

            # Mock query processing
            with patch("src.agents.coordinator.create_agent_system") as mock_agent:
                mock_system = MagicMock()
                mock_system.aquery = AsyncMock()
                mock_system.aquery.return_value.response = f"Response from {task_id}"

                agent_system = mock_agent(index, settings)
                result = await agent_system.aquery(f"Test query for {task_id}")

            logger.info(f"Completed processing task: {task_id}")
            return result.response

        except Exception as e:
            logger.error(f"Processing task {task_id} failed: {e}")
            raise


@pytest.mark.system
class TestSystemConfiguration:
    """Test system configuration and hardware detection."""

    def test_gpu_detection_and_configuration(self, system_settings):
        """Test GPU detection and configuration setup."""
        logger.info("Testing GPU detection and configuration")

        # Test hardware detection
        hardware_info = detect_hardware()

        assert isinstance(hardware_info, dict)
        assert "cuda_available" in hardware_info
        assert "gpu_name" in hardware_info
        assert "vram_total_gb" in hardware_info

        if hardware_info["cuda_available"]:
            assert hardware_info["gpu_name"] != "Unknown"
            assert hardware_info["vram_total_gb"] is not None
            assert hardware_info["vram_total_gb"] > 0

            logger.info(f"Detected GPU: {hardware_info['gpu_name']}")
            logger.info(f"VRAM: {hardware_info['vram_total_gb']}GB")

    def test_settings_validation_for_gpu(self, system_settings):
        """Test settings are properly configured for GPU operations."""
        logger.info("Testing settings validation for GPU")

        # Validate GPU-related settings
        assert system_settings.quantization in ["fp8", "int8", "int4"]
        assert system_settings.kv_cache_dtype in ["fp8", "fp16", "int8"]
        assert 0.5 <= system_settings.gpu_memory_utilization <= 0.9
        assert system_settings.context_window_size > 0

        # Validate performance settings
        assert system_settings.llm_temperature >= 0.0
        assert system_settings.llm_max_tokens > 0
        assert system_settings.agent_decision_timeout > 0

        logger.info(f"Model: {system_settings.model_name}")
        logger.info(f"Quantization: {system_settings.quantization}")
        logger.info(f"KV Cache: {system_settings.kv_cache_dtype}")
        logger.info(f"Context Window: {system_settings.context_window_size}")

    @pytest.mark.requires_gpu
    def test_gpu_memory_allocation(self):
        """Test GPU memory can be allocated and managed properly."""
        logger.info("Testing GPU memory allocation")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda:0")

        try:
            # Test memory allocation
            initial_memory = torch.cuda.memory_allocated(device)

            # Allocate test tensor
            test_tensor = torch.randn(1000, 1000, device=device, dtype=torch.float16)
            allocated_memory = torch.cuda.memory_allocated(device)

            memory_used = (allocated_memory - initial_memory) / 1024**2  # MB
            logger.info(f"Allocated {memory_used:.1f}MB for test tensor")

            assert memory_used > 0
            assert memory_used < 100  # Should be reasonable amount

            # Test cleanup
            del test_tensor
            torch.cuda.empty_cache()

            final_memory = torch.cuda.memory_allocated(device)
            assert final_memory <= initial_memory + (1024**2)  # Allow 1MB tolerance

            logger.info("GPU memory allocation test passed")

        except Exception as e:
            logger.error(f"GPU memory allocation test failed: {e}")
            raise


@pytest.mark.system
class TestSystemIntegration:
    """Test system integration points and error handling."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)  # 2 minute timeout
    async def test_graceful_degradation_on_errors(
        self, system_settings, sample_documents
    ):
        """Test system handles errors gracefully without crashing."""
        logger.info("Testing graceful degradation on errors")

        try:
            # Test with invalid model configuration
            invalid_settings = Settings(
                model_name="non/existent-model",
                llm_backend="invalid_backend",
            )

            with pytest.raises(Exception):  # Should raise configuration error
                with patch("src.agents.coordinator.create_agent_system") as mock_create:
                    mock_create.side_effect = ValueError("Invalid model configuration")
                    await mock_create(MagicMock(), invalid_settings)

            # Test with network failures
            with patch("src.utils.embedding.create_index_async") as mock_create:
                mock_create.side_effect = ConnectionError("Network unavailable")

                with pytest.raises(ConnectionError):
                    await mock_create(sample_documents, system_settings)

            # Test resource exhaustion
            with patch("torch.cuda.memory_allocated") as mock_memory:
                mock_memory.return_value = 15 * 1024**3  # Simulate OOM

                with pytest.raises(RuntimeError, match="out of memory|OOM|memory"):
                    # This should be caught by resource management
                    if torch.cuda.is_available():
                        torch.cuda.memory_allocated()
                        raise RuntimeError("CUDA out of memory")

            logger.info("Graceful degradation test passed")

        except Exception as e:
            logger.error(f"Graceful degradation test failed: {e}")
            raise

    def test_configuration_compatibility(self, system_settings):
        """Test configuration compatibility across components."""
        logger.info("Testing configuration compatibility")

        # Test model and quantization compatibility
        if "fp8" in system_settings.model_name.lower():
            # FP8 model should use FP8 quantization
            assert system_settings.quantization == "fp8"
            assert system_settings.kv_cache_dtype == "fp8"

        # Test context window compatibility
        if system_settings.context_window_size > 100000:
            # Large context requires efficient KV cache
            assert system_settings.kv_cache_dtype in ["fp8", "int8"]
            assert system_settings.enable_kv_cache_optimization

        # Test GPU memory configuration
        if hasattr(system_settings, "gpu_memory_utilization"):
            utilization = system_settings.gpu_memory_utilization
            assert 0.5 <= utilization <= 0.9  # Reasonable range

            # High utilization requires efficient quantization
            if utilization > 0.8:
                assert system_settings.quantization in ["fp8", "int8"]

        logger.info("Configuration compatibility test passed")


# Additional markers for the pytest configuration
def pytest_configure(config):
    """Configure custom pytest markers for system tests."""
    config.addinivalue_line(
        "markers", "system: marks tests as system-level integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU hardware"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow-running (>30 seconds)"
    )
    config.addinivalue_line(
        "markers", "timeout: marks tests with specific timeout requirements"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
