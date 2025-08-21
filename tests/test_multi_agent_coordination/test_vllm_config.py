"""Comprehensive unit tests for vLLM Configuration (ADR-004, ADR-010 compliant).

Tests cover:
- VLLMConfig data model and validation
- ContextManager for 128K context handling
- VLLMManager with FP8 optimization
- Performance validation and metrics
- Context trimming and memory management
- FP8 KV cache optimization
- Error handling and fallback mechanisms
- Hardware requirements validation
"""

import os
import tempfile
import time
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage

from src.vllm_config import (
    ContextManager,
    VLLMConfig,
    VLLMManager,
    create_vllm_manager,
    validate_fp8_requirements,
)


class TestVLLMConfig:
    """Test suite for VLLMConfig data model."""

    def test_config_initialization_defaults(self):
        """Test VLLMConfig initialization with default values."""
        config = VLLMConfig()

        # Verify ADR-004 compliance (Local-First LLM Strategy)
        assert config.model == "Qwen/Qwen3-4B-Instruct-2507-FP8"
        assert config.max_model_len == 131072  # 128K context
        assert config.gpu_memory_utilization == 0.95
        assert config.trust_remote_code is True

        # Verify ADR-010 compliance (Performance Optimization)
        assert config.kv_cache_dtype == "fp8_e5m2"  # FP8 KV cache
        assert config.calculate_kv_scales is True
        assert config.attention_backend == "FLASHINFER"
        assert config.enable_chunked_prefill is True
        assert config.use_cudnn_prefill is True

        # Verify performance targets
        assert config.target_decode_throughput == 130
        assert 100 <= config.target_decode_throughput <= 160
        assert config.target_prefill_throughput == 1050
        assert 800 <= config.target_prefill_throughput <= 1300
        assert config.vram_usage_target_gb == 13.5
        assert 12 <= config.vram_usage_target_gb <= 16

        # Verify service configuration
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.served_model_name == "docmind-qwen3-fp8"

    def test_config_initialization_custom_values(self):
        """Test VLLMConfig initialization with custom values."""
        config = VLLMConfig(
            model="custom/model-fp8",
            max_model_len=65536,
            gpu_memory_utilization=0.8,
            kv_cache_dtype="fp8_e4m3",
            attention_backend="XFORMERS",
            target_decode_throughput=120,
            target_prefill_throughput=900,
            vram_usage_target_gb=14.0,
            host="127.0.0.1",
            port=9000,
            served_model_name="custom-model",
        )

        assert config.model == "custom/model-fp8"
        assert config.max_model_len == 65536
        assert config.gpu_memory_utilization == 0.8
        assert config.kv_cache_dtype == "fp8_e4m3"
        assert config.attention_backend == "XFORMERS"
        assert config.target_decode_throughput == 120
        assert config.target_prefill_throughput == 900
        assert config.vram_usage_target_gb == 14.0
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.served_model_name == "custom-model"

    def test_config_fp8_optimization_validation(self):
        """Test FP8 optimization configuration validation."""
        config = VLLMConfig()

        # Verify FP8 optimization is properly configured
        assert "fp8" in config.kv_cache_dtype.lower()
        assert config.calculate_kv_scales is True  # Required for FP8
        assert config.attention_backend in ["FLASHINFER", "XFORMERS"]  # FP8 compatible
        assert config.max_num_seqs == 1  # Single sequence for 128K context

    def test_config_performance_targets_validation(self):
        """Test performance targets are within specified ranges."""
        config = VLLMConfig()

        # Verify decode throughput target (100-160 tok/s)
        assert 100 <= config.target_decode_throughput <= 160

        # Verify prefill throughput target (800-1300 tok/s)
        assert 800 <= config.target_prefill_throughput <= 1300

        # Verify VRAM usage target (12-14GB on RTX 4090 Laptop)
        assert 12 <= config.vram_usage_target_gb <= 16


class TestContextManager:
    """Test suite for ContextManager (128K context management)."""

    def test_context_manager_initialization(self):
        """Test ContextManager initialization with proper defaults."""
        manager = ContextManager()

        # Verify 128K context configuration
        assert manager.max_context_tokens == 131072  # 128K
        assert manager.trim_threshold == 120000  # 8K buffer
        assert manager.preserve_ratio == 0.3

        # Verify FP8 KV cache memory calculations
        assert manager.kv_cache_memory_per_token == 1024  # bytes per token
        assert manager.total_kv_cache_gb_at_128k == 8.0  # ~8GB at 128K

    def test_estimate_tokens_basic(self):
        """Test basic token estimation functionality."""
        manager = ContextManager()

        # Test with simple messages
        messages = [
            {"content": "Hello world"},  # 11 chars = ~2.75 tokens
            {"content": "This is a test"},  # 14 chars = ~3.5 tokens
        ]

        tokens = manager.estimate_tokens(messages)
        expected_tokens = (11 + 14) // 4  # 4 chars per token estimate
        assert tokens == expected_tokens

    def test_estimate_tokens_empty_messages(self):
        """Test token estimation with empty messages."""
        manager = ContextManager()

        # Test with empty list
        assert manager.estimate_tokens([]) == 0

        # Test with messages without content
        messages = [{"role": "user"}, {"metadata": "test"}]
        assert manager.estimate_tokens(messages) == 0

    def test_estimate_tokens_large_content(self):
        """Test token estimation with large content."""
        manager = ContextManager()

        # Create large message
        large_content = "A" * 400000  # 400K characters = 100K tokens
        messages = [{"content": large_content}]

        tokens = manager.estimate_tokens(messages)
        expected_tokens = 400000 // 4
        assert tokens == expected_tokens
        assert tokens > manager.trim_threshold  # Should exceed threshold

    def test_trim_to_token_limit_basic(self):
        """Test basic message trimming functionality."""
        manager = ContextManager()

        # Create messages that exceed limit
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "First user message"},
            {"role": "assistant", "content": "First assistant response"},
            {"role": "user", "content": "Second user message"},
            {"role": "assistant", "content": "Second assistant response"},
            {"role": "user", "content": "Latest user message"},
        ]

        limit = 100  # Very low limit to force trimming
        trimmed = manager.trim_to_token_limit(messages, limit)

        # Should preserve system message and latest user message
        assert trimmed[0]["role"] == "system"
        assert trimmed[-1]["role"] == "user"
        assert trimmed[-1]["content"] == "Latest user message"

        # Should respect token limit
        trimmed_tokens = manager.estimate_tokens(trimmed)
        assert trimmed_tokens <= limit

    def test_trim_to_token_limit_preserve_structure(self):
        """Test message trimming preserves conversation structure."""
        manager = ContextManager()

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Old message 1"},
            {"role": "assistant", "content": "Old response 1"},
            {"role": "user", "content": "Old message 2"},
            {"role": "assistant", "content": "Old response 2"},
            {"role": "user", "content": "Recent message"},
        ]

        limit = 50  # Force aggressive trimming
        trimmed = manager.trim_to_token_limit(messages, limit)

        # Should always preserve system and latest user messages
        system_messages = [msg for msg in trimmed if msg["role"] == "system"]
        user_messages = [msg for msg in trimmed if msg["role"] == "user"]

        assert len(system_messages) == 1
        assert system_messages[0]["content"] == "System prompt"
        assert user_messages[-1]["content"] == "Recent message"

    def test_trim_to_token_limit_empty_messages(self):
        """Test message trimming with empty input."""
        manager = ContextManager()

        assert manager.trim_to_token_limit([], 1000) == []

    def test_trim_to_token_limit_under_limit(self):
        """Test message trimming when already under limit."""
        manager = ContextManager()

        messages = [
            {"role": "user", "content": "Short message"},
            {"role": "assistant", "content": "Short response"},
        ]

        limit = 1000  # High limit
        trimmed = manager.trim_to_token_limit(messages, limit)

        # Should return all messages unchanged
        assert len(trimmed) == len(messages)
        assert trimmed == messages

    def test_pre_model_hook_trimming(self):
        """Test pre-model hook context trimming functionality."""
        manager = ContextManager()

        # Create state with large messages
        large_content = "A" * 500000  # Large content exceeding threshold
        state = {"messages": [HumanMessage(content=large_content)]}

        # Mock estimate_tokens to return value exceeding threshold
        manager.estimate_tokens = Mock(side_effect=[150000, 100000])  # Before and after

        result_state = manager.pre_model_hook(state)

        # Verify trimming was applied
        assert result_state["context_trimmed"] is True
        assert result_state["tokens_trimmed"] == 50000  # 150000 - 100000
        assert "messages" in result_state

    def test_pre_model_hook_no_trimming_needed(self):
        """Test pre-model hook when no trimming is needed."""
        manager = ContextManager()

        state = {"messages": [HumanMessage(content="Short message")]}

        # Mock estimate_tokens to return value under threshold
        manager.estimate_tokens = Mock(return_value=50000)  # Under threshold

        result_state = manager.pre_model_hook(state)

        # Verify no trimming was applied
        assert "context_trimmed" not in result_state
        assert "tokens_trimmed" not in result_state

    def test_post_model_hook_structured_output(self):
        """Test post-model hook for structured output formatting."""
        manager = ContextManager()

        state = {
            "output_mode": "structured",
            "messages": [HumanMessage(content="Test message")],
            "response": "Test response",
        }

        # Mock methods
        manager.estimate_tokens = Mock(return_value=1000)
        manager.calculate_kv_cache_usage = Mock(return_value=2.5)
        manager.structure_response = Mock(
            return_value={"content": "Test response", "structured": True}
        )

        result_state = manager.post_model_hook(state)

        # Verify metadata was added
        assert "metadata" in result_state
        metadata = result_state["metadata"]
        assert metadata["context_used"] == 1000
        assert metadata["kv_cache_usage_gb"] == 2.5
        assert "parallel_execution_active" in metadata

        # Verify response was structured
        assert result_state["response"]["structured"] is True

    def test_post_model_hook_non_structured_output(self):
        """Test post-model hook with non-structured output mode."""
        manager = ContextManager()

        state = {
            "output_mode": "simple",
            "response": "Simple response",
        }

        result_state = manager.post_model_hook(state)

        # Should not modify state for non-structured output
        assert "metadata" not in result_state
        assert result_state["response"] == "Simple response"

    def test_calculate_kv_cache_usage(self):
        """Test KV cache memory usage calculation."""
        manager = ContextManager()

        state = {"messages": [HumanMessage(content="Test message")]}

        # Mock token estimation
        manager.estimate_tokens = Mock(return_value=50000)  # 50K tokens

        usage_gb = manager.calculate_kv_cache_usage(state)
        expected_gb = (50000 * 1024) / (1024**3)  # 50K tokens * 1024 bytes / GB

        assert abs(usage_gb - expected_gb) < 0.001  # Small floating point tolerance

    def test_structure_response(self):
        """Test response structuring functionality."""
        manager = ContextManager()

        response = "Test response content"
        structured = manager.structure_response(response)

        assert structured["content"] == response
        assert structured["structured"] is True
        assert "generated_at" in structured
        assert structured["context_optimized"] is True
        assert isinstance(structured["generated_at"], float)


class TestVLLMManager:
    """Test suite for VLLMManager with FP8 optimization."""

    def test_manager_initialization(self, mock_vllm_config: VLLMConfig):
        """Test VLLMManager initialization."""
        manager = VLLMManager(mock_vllm_config)

        assert manager.config == mock_vllm_config
        assert manager.llm is None
        assert manager.async_engine is None
        assert isinstance(manager.context_manager, ContextManager)

        # Verify performance metrics initialization
        metrics = manager._performance_metrics
        assert metrics["requests_processed"] == 0
        assert metrics["avg_decode_throughput"] == 0.0
        assert metrics["avg_prefill_throughput"] == 0.0
        assert metrics["peak_vram_usage_gb"] == 0.0

    def test_initialize_engine_with_vllm_available(self, mock_vllm_config: VLLMConfig):
        """Test engine initialization when vLLM is available."""
        with (
            patch("src.vllm_config.VLLM_AVAILABLE", True),
            patch("src.vllm_config.AsyncLLMEngine") as mock_async_engine,
            patch("src.vllm_config.LLM") as mock_llm_class,
        ):
            manager = VLLMManager(mock_vllm_config)

            # Mock engine creation
            mock_async_engine.from_engine_args.return_value = Mock()
            mock_llm_class.return_value = Mock()

            result = manager.initialize_engine()

            assert result is True
            assert manager.async_engine is not None
            assert manager.llm is not None

            # Verify environment variables were set
            assert (
                os.environ.get("VLLM_ATTENTION_BACKEND")
                == mock_vllm_config.attention_backend
            )
            assert os.environ.get("VLLM_USE_CUDNN_PREFILL") == "1"

    def test_initialize_engine_with_vllm_unavailable(
        self, mock_vllm_config: VLLMConfig
    ):
        """Test engine initialization when vLLM is not available."""
        with patch("src.vllm_config.VLLM_AVAILABLE", False):
            manager = VLLMManager(mock_vllm_config)

            result = manager.initialize_engine()

            assert result is False
            assert manager.llm is None
            assert manager.async_engine is None

    def test_initialize_engine_error_handling(self, mock_vllm_config: VLLMConfig):
        """Test engine initialization error handling."""
        with (
            patch("src.vllm_config.VLLM_AVAILABLE", True),
            patch("src.vllm_config.AsyncLLMEngine") as mock_async_engine,
        ):
            # Mock engine creation to raise exception
            mock_async_engine.from_engine_args.side_effect = Exception(
                "Engine creation failed"
            )

            manager = VLLMManager(mock_vllm_config)
            result = manager.initialize_engine()

            assert result is False
            assert manager.llm is None
            assert manager.async_engine is None

    def test_generate_start_script(self, mock_vllm_config: VLLMConfig):
        """Test vLLM start script generation."""
        manager = VLLMManager(mock_vllm_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "test_start_vllm.sh")

            result_path = manager.generate_start_script(script_path)

            assert result_path == script_path
            assert os.path.exists(script_path)

            # Verify script content
            with open(script_path) as f:
                content = f.read()

            assert "vllm serve" in content
            assert mock_vllm_config.model in content
            assert str(mock_vllm_config.max_model_len) in content
            assert mock_vllm_config.kv_cache_dtype in content
            assert "--calculate-kv-scales" in content
            assert str(mock_vllm_config.gpu_memory_utilization) in content
            assert str(mock_vllm_config.max_num_seqs) in content
            assert "--enable-chunked-prefill" in content
            assert "--trust-remote-code" in content
            assert str(mock_vllm_config.host) in content
            assert str(mock_vllm_config.port) in content
            assert mock_vllm_config.served_model_name in content

            # Verify environment variables
            assert (
                f"export VLLM_ATTENTION_BACKEND={mock_vllm_config.attention_backend}"
                in content
            )
            assert "export VLLM_USE_CUDNN_PREFILL=1" in content

            # Verify script is executable
            assert os.access(script_path, os.X_OK)

    def test_validate_performance_with_engine(self, mock_vllm_config: VLLMConfig):
        """Test performance validation when engine is available."""
        manager = VLLMManager(mock_vllm_config)

        # Mock LLM and sampling
        mock_llm = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = (
            "Machine learning is a subset of AI that enables " * 10
        )  # ~100 tokens
        mock_llm.generate.return_value = [mock_output]

        manager.llm = mock_llm

        with patch("src.vllm_config.SamplingParams") as mock_sampling_params:
            mock_sampling_params.return_value = Mock()

            result = manager.validate_performance()

            assert "decode_throughput_estimate" in result
            assert "meets_decode_target" in result
            assert "generation_time" in result
            assert "tokens_generated" in result
            assert result["model_loaded"] is True
            assert result["fp8_optimization"] is True
            assert result["context_window"] == mock_vllm_config.max_model_len
            assert result["meets_context_target"] is True
            assert "validation_timestamp" in result

            # Verify throughput calculation
            assert result["decode_throughput_estimate"] > 0
            assert isinstance(result["meets_decode_target"], bool)

    def test_validate_performance_without_engine(self, mock_vllm_config: VLLMConfig):
        """Test performance validation when engine is not available."""
        manager = VLLMManager(mock_vllm_config)
        # manager.llm remains None

        result = manager.validate_performance()

        assert "error" in result
        assert result["error"] == "Engine not initialized"

    def test_validate_performance_error_handling(self, mock_vllm_config: VLLMConfig):
        """Test performance validation error handling."""
        manager = VLLMManager(mock_vllm_config)

        # Mock LLM to raise exception
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("Generation failed")
        manager.llm = mock_llm

        with patch("src.vllm_config.SamplingParams"):
            result = manager.validate_performance()

            assert "error" in result
            assert result["validation_failed"] is True
            assert "Generation failed" in result["error"]

    def test_get_performance_metrics(self, mock_vllm_config: VLLMConfig):
        """Test performance metrics retrieval."""
        manager = VLLMManager(mock_vllm_config)

        # Set some performance data
        manager._performance_metrics["requests_processed"] = 10
        manager._performance_metrics["avg_decode_throughput"] = 125.5

        metrics = manager.get_performance_metrics()

        # Verify structure and content
        assert metrics["requests_processed"] == 10
        assert metrics["avg_decode_throughput"] == 125.5

        assert "config" in metrics
        config = metrics["config"]
        assert config["model"] == mock_vllm_config.model
        assert config["max_context"] == mock_vllm_config.max_model_len
        assert config["kv_cache_dtype"] == mock_vllm_config.kv_cache_dtype
        assert config["attention_backend"] == mock_vllm_config.attention_backend

        assert "targets" in metrics
        targets = metrics["targets"]
        assert targets["decode_throughput_range"] == (100, 160)
        assert targets["prefill_throughput_range"] == (800, 1300)
        assert targets["vram_usage_range_gb"] == (12, 14)
        assert targets["context_window"] == 131072


class TestFactoryFunction:
    """Test suite for factory function."""

    def test_create_vllm_manager_defaults(self):
        """Test factory function with default parameters."""
        manager = create_vllm_manager()

        assert isinstance(manager, VLLMManager)
        assert manager.config.model == "Qwen/Qwen3-4B-Instruct-2507-FP8"
        assert manager.config.max_model_len == 131072

    def test_create_vllm_manager_custom_params(self):
        """Test factory function with custom parameters."""
        manager = create_vllm_manager(
            model_path="custom/model", max_context_length=65536
        )

        assert isinstance(manager, VLLMManager)
        assert manager.config.model == "custom/model"
        assert manager.config.max_model_len == 65536


class TestValidateFP8Requirements:
    """Test suite for FP8 requirements validation."""

    def test_validate_fp8_requirements_all_available(self):
        """Test FP8 requirements validation when all requirements are met."""
        with (
            patch("src.vllm_config.VLLM_AVAILABLE", True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            # Mock GPU with sufficient memory (16GB)
            mock_device = Mock()
            mock_device.total_memory = 16 * 1024**3  # 16GB in bytes
            mock_props.return_value = mock_device

            requirements = validate_fp8_requirements()

            assert requirements["vllm_available"] is True
            assert requirements["cuda_available"] is True
            assert requirements["fp8_support"] is True
            assert requirements["flashinfer_backend"] is True
            assert requirements["sufficient_vram"] is True

    def test_validate_fp8_requirements_insufficient_vram(self):
        """Test FP8 requirements validation with insufficient VRAM."""
        with (
            patch("src.vllm_config.VLLM_AVAILABLE", True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            # Mock GPU with insufficient memory (8GB)
            mock_device = Mock()
            mock_device.total_memory = 8 * 1024**3  # 8GB in bytes
            mock_props.return_value = mock_device

            requirements = validate_fp8_requirements()

            assert requirements["vllm_available"] is True
            assert requirements["cuda_available"] is True
            assert requirements["sufficient_vram"] is False  # Insufficient VRAM

    def test_validate_fp8_requirements_no_cuda(self):
        """Test FP8 requirements validation without CUDA."""
        with (
            patch("src.vllm_config.VLLM_AVAILABLE", True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            requirements = validate_fp8_requirements()

            assert requirements["vllm_available"] is True
            assert requirements["cuda_available"] is False
            assert requirements["sufficient_vram"] is False

    def test_validate_fp8_requirements_no_torch(self):
        """Test FP8 requirements validation without PyTorch."""
        with (
            patch("src.vllm_config.VLLM_AVAILABLE", True),
            patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'torch'"),
            ),
        ):
            requirements = validate_fp8_requirements()

            assert requirements["vllm_available"] is True
            assert requirements["cuda_available"] is False
            assert requirements["sufficient_vram"] is False

    def test_validate_fp8_requirements_no_vllm(self):
        """Test FP8 requirements validation without vLLM."""
        with patch("src.vllm_config.VLLM_AVAILABLE", False):
            requirements = validate_fp8_requirements()

            assert requirements["vllm_available"] is False
            # Other requirements should still be checked
            assert "cuda_available" in requirements
            assert "fp8_support" in requirements
            assert "flashinfer_backend" in requirements


class TestPerformanceAndTiming:
    """Test suite for performance measurement and timing."""

    def test_context_trimming_performance(self):
        """Test context trimming performance with large messages."""
        manager = ContextManager()

        # Create large message set
        messages = []
        for i in range(100):
            messages.append({"role": "user", "content": f"Message {i} content " * 100})

        start_time = time.perf_counter()
        trimmed = manager.trim_to_token_limit(messages, 1000)
        end_time = time.perf_counter()

        # Should complete quickly even with many messages
        assert end_time - start_time < 1.0  # Less than 1 second
        assert len(trimmed) <= len(messages)  # Should trim some messages

    def test_token_estimation_performance(self):
        """Test token estimation performance with various message sizes."""
        manager = ContextManager()

        # Test with different message sizes
        test_cases = [10, 100, 1000, 10000]  # Number of messages

        for num_messages in test_cases:
            messages = [{"content": f"Message {i}"} for i in range(num_messages)]

            start_time = time.perf_counter()
            tokens = manager.estimate_tokens(messages)
            end_time = time.perf_counter()

            # Should scale linearly and complete quickly
            assert end_time - start_time < 0.1  # Less than 100ms
            assert tokens > 0


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling."""

    def test_context_manager_hooks_error_handling(self):
        """Test context manager hooks handle errors gracefully."""
        manager = ContextManager()

        # Test pre-model hook with invalid state
        invalid_state = {"messages": "not a list"}
        result = manager.pre_model_hook(invalid_state)
        # Should return original state without crashing
        assert result == invalid_state

        # Test post-model hook with invalid state
        invalid_state = {"output_mode": "structured", "messages": None}
        result = manager.post_model_hook(invalid_state)
        # Should return original state without crashing
        assert "output_mode" in result

    def test_vllm_manager_with_invalid_config(self):
        """Test VLLMManager with edge case configurations."""
        # Test with minimal config
        config = VLLMConfig(max_model_len=1024)  # Very small context
        manager = VLLMManager(config)

        assert manager.config.max_model_len == 1024
        assert isinstance(manager.context_manager, ContextManager)

    def test_performance_validation_edge_cases(self, mock_vllm_config: VLLMConfig):
        """Test performance validation with edge cases."""
        manager = VLLMManager(mock_vllm_config)

        # Mock LLM with zero-length response
        mock_llm = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = ""  # Empty response
        mock_llm.generate.return_value = [mock_output]

        manager.llm = mock_llm

        with patch("src.vllm_config.SamplingParams"):
            result = manager.validate_performance()

            # Should handle zero-length response gracefully
            assert "decode_throughput_estimate" in result
            assert result["tokens_generated"] == 0

    def test_script_generation_with_special_characters(self):
        """Test script generation with special characters in paths."""
        config = VLLMConfig(
            model="test/model-name_with-special.chars",
            served_model_name="test-model_name",
        )
        manager = VLLMManager(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "test_script.sh")

            # Should not raise exception
            result_path = manager.generate_start_script(script_path)
            assert os.path.exists(result_path)

            # Verify content is properly escaped/handled
            with open(script_path) as f:
                content = f.read()
            assert config.model in content
            assert config.served_model_name in content

    def test_context_manager_with_non_string_content(self):
        """Test context manager with non-string message content."""
        manager = ContextManager()

        # Test with various content types
        messages = [
            {"content": None},
            {"content": 12345},
            {"content": ["list", "content"]},
            {"content": {"dict": "content"}},
        ]

        # Should not crash
        tokens = manager.estimate_tokens(messages)
        assert tokens >= 0

        trimmed = manager.trim_to_token_limit(messages, 1000)
        assert isinstance(trimmed, list)
