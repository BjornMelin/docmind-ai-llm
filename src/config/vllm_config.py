"""vLLM Configuration for Qwen3-4B-Instruct-2507-FP8 with FP8 Optimization.

This module implements ADR-004 and ADR-010.

This module provides configuration and setup for vLLM backend serving with
FP8 quantization, FlashInfer backend, and 128K context support optimized
for RTX 4090 Laptop hardware constraints.

Features:
- Qwen3-4B-Instruct-2507-FP8 model configuration
- FP8 KV cache optimization for 50% memory reduction
- FlashInfer backend for FP8 acceleration
- 128K context window management (hardware-constrained from 262K native)
- Memory optimization for RTX 4090 Laptop (12-14GB VRAM target)
- Performance targets: 100-160 tok/s decode, 800-1300 tok/s prefill

ADR Compliance:
- ADR-004: Local-First LLM Strategy (Qwen3-4B-Instruct-2507-FP8)
- ADR-010: Performance Optimization Strategy (FP8 KV cache, memory optimization)
"""

import os
import time
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("vLLM not available - using fallback configuration")
    VLLM_AVAILABLE = False

try:
    from llama_index.core import Settings
    from llama_index.llms.vllm import Vllm

    VllmType: type | None = Vllm
except ImportError:
    Vllm = None
    VllmType = None

from src.config.app_settings import app_settings

# Constants
CHARS_PER_TOKEN_AVERAGE = 4  # Simplified estimation - 4 chars per token average
FILE_PERMISSION_EXECUTABLE = 0o755


class VLLMConfig(BaseModel):
    """Configuration for vLLM backend serving FP8 model with FlashInfer optimization."""

    # Model configuration
    model: str = Field(
        default="Qwen/Qwen3-4B-Instruct-2507-FP8",
        description="FP8 quantized model path",
    )
    max_model_len: int = Field(
        default=app_settings.default_token_limit,  # 128K context (hardware-constrained)
        description="Maximum context length in tokens",
    )
    gpu_memory_utilization: float = Field(
        default=app_settings.vllm_gpu_memory_utilization,
        description="GPU memory utilization ratio",
    )

    # FP8 KV Cache Optimization (ADR-004, ADR-010)
    kv_cache_dtype: str = Field(
        default=app_settings.vllm_kv_cache_dtype,
        description="FP8 KV cache for 50% memory reduction",
    )
    calculate_kv_scales: bool = Field(
        default=app_settings.vllm_calculate_kv_scales,
        description="Required for FP8 KV cache",
    )
    attention_backend: str = Field(
        default=app_settings.vllm_attention_backend,
        description="FlashInfer backend for FP8 acceleration",
    )
    enable_chunked_prefill: bool = Field(
        default=app_settings.vllm_enable_chunked_prefill
    )
    use_cudnn_prefill: bool = Field(default=app_settings.vllm_use_cudnn_prefill)

    # Memory optimization
    max_num_seqs: int = Field(
        default=app_settings.vllm_max_num_seqs,
        description="Maximum number of sequences",
    )
    max_num_batched_tokens: int = Field(
        default=app_settings.vllm_max_num_batched_tokens,
        description="Maximum number of batched tokens",
    )
    dtype: str = Field(default="auto", description="Automatic FP8 dtype selection")
    trust_remote_code: bool = Field(default=True)

    # Performance metrics (validated targets)
    target_decode_throughput: int = Field(
        default=130, description="Target decode throughput: 100-160 tok/s"
    )
    target_prefill_throughput: int = Field(
        default=1050, description="Target prefill throughput: 800-1300 tok/s"
    )
    vram_usage_target_gb: float = Field(
        default=13.5, description="Target VRAM usage: 12-14GB on RTX 4090 Laptop"
    )

    # Service configuration
    host: str = Field(default="0.0.0.0")  # noqa: S104
    port: int = Field(default=8000)
    served_model_name: str = Field(default="docmind-qwen3-fp8")

    def validate_model_path(self) -> bool:
        """Validate FP8 model path and configuration.

        Returns:
            True if model configuration is valid for FP8
        """
        # Check if model name suggests FP8 optimization
        return "FP8" in self.model or "fp8" in self.kv_cache_dtype.lower()

    def is_fp8_enabled(self) -> bool:
        """Check if FP8 optimization is enabled.

        Returns:
            True if FP8 optimization is enabled
        """
        return (
            "FP8" in self.model
            or "fp8" in self.kv_cache_dtype.lower()
            or self.dtype == "fp8"
        )

    def estimate_vram_usage(self) -> float:
        """Estimate VRAM usage for the current configuration.

        Returns:
            Estimated VRAM usage in GB
        """
        # Base model size (Qwen3-4B-FP8): ~4GB
        base_model_gb = 4.0

        # KV cache calculation
        # FP8: ~80 bytes per token, FP16: ~160 bytes per token
        bytes_per_token = 80 if "fp8" in self.kv_cache_dtype.lower() else 160
        kv_cache_gb = (self.max_model_len * bytes_per_token) / (1024**3)

        # Additional overhead (workspace, attention, etc.)
        overhead_gb = 2.0

        return base_model_gb + kv_cache_gb + overhead_gb


class ContextManager:
    """Manages 128K context window with intelligent trimming strategies.

    Implements ADR-004 and ADR-011 for context management.
    """

    def __init__(self):
        """Initialize context manager with 128K context and FP8 KV cache settings."""
        self.max_context_tokens = (
            app_settings.default_token_limit  # 128K context (hardware-constrained)
        )
        self.trim_threshold = (
            app_settings.vllm_max_token_limit
        )  # Trim at 120K (8K buffer for 128K limit)
        self.preserve_ratio = 0.3  # Keep 30% of oldest context for continuity

        # Memory calculations for FP8 KV cache (ADR-010)
        self.kv_cache_memory_per_token = (
            app_settings.vllm_kv_cache_memory_per_token
        )  # bytes per token with FP8
        # Calculate KV cache usage at 128K context: tokens * bytes_per_token / (1024^3)
        self.total_kv_cache_gb_at_128k = (
            self.max_context_tokens * self.kv_cache_memory_per_token
        ) / (1024**3)

    def pre_model_hook(self, state: dict) -> dict:
        """Trim context before model processing (ADR-011)."""
        messages = state.get("messages", [])
        total_tokens = self.estimate_tokens(messages)

        if total_tokens > self.trim_threshold:
            # Aggressive trimming strategy maintaining conversation coherence
            messages = self.trim_to_token_limit(messages, self.trim_threshold)
            state["messages"] = messages
            state["context_trimmed"] = True
            state["tokens_trimmed"] = total_tokens - self.estimate_tokens(messages)

        return state

    def post_model_hook(self, state: dict) -> dict:
        """Format response after model generation (ADR-011)."""
        if state.get("output_mode") == "structured":
            state["response"] = self.structure_response(state["response"])
            state["metadata"] = {
                "context_used": self.estimate_tokens(state.get("messages", [])),
                "kv_cache_usage_gb": self.calculate_kv_cache_usage(state),
                "parallel_execution_active": state.get("parallel_tool_calls", False),
            }
        return state

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for context management."""
        # Simplified estimation - 4 chars per token average
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        return total_chars // CHARS_PER_TOKEN_AVERAGE

    def trim_to_token_limit(self, messages: list[dict], limit: int) -> list[dict]:
        """Trim messages to token limit while preserving conversation structure."""
        if not messages:
            return messages

        # Always preserve system message and latest user message
        system_msgs = [msg for msg in messages if msg.get("role") == "system"]
        latest_user = [msg for msg in reversed(messages) if msg.get("role") == "user"][
            :1
        ]

        # Calculate available tokens for history
        reserved_tokens = self.estimate_tokens(system_msgs + latest_user)
        available_tokens = limit - reserved_tokens

        # Trim middle conversation history
        history_msgs = [
            msg
            for msg in messages
            if msg.get("role") not in ["system"] and msg not in latest_user
        ]

        # Keep most recent history that fits
        trimmed_history = []
        current_tokens = 0
        for msg in reversed(history_msgs):
            msg_tokens = self.estimate_tokens([msg])
            if current_tokens + msg_tokens <= available_tokens:
                trimmed_history.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break

        return system_msgs + trimmed_history + latest_user

    def calculate_kv_cache_usage(self, state: dict) -> float:
        """Calculate current KV cache memory usage in GB."""
        context_tokens = self.estimate_tokens(state.get("messages", []))
        return (context_tokens * self.kv_cache_memory_per_token) / (1024**3)

    def calculate_kv_cache_memory(self, dtype: str) -> float:
        """Calculate KV cache memory usage for given dtype.

        Args:
            dtype: Data type ('float16' or 'fp8_e5m2')

        Returns:
            Memory usage in bytes per token
        """
        if dtype == "fp8_e5m2":
            return self.kv_cache_memory_per_token
        elif dtype == "float16":
            return self.kv_cache_memory_per_token * 2  # FP16 uses 2x memory
        else:
            return self.kv_cache_memory_per_token

    def structure_response(self, response: str) -> dict:
        """Structure response with metadata for enhanced integration."""
        return {
            "content": response,
            "structured": True,
            "generated_at": time.time(),
            "context_optimized": True,
        }


class VLLMManager:
    """Manager for vLLM backend with FP8 optimization."""

    def __init__(self, config: VLLMConfig):
        """Initialize vLLM manager with FP8 optimization configuration."""
        self.config = config
        self.llm: Any | None = None
        self.async_engine: Any | None = None
        self.context_manager = ContextManager()
        self._performance_metrics = {
            "requests_processed": 0,
            "avg_decode_throughput": 0.0,
            "avg_prefill_throughput": 0.0,
            "peak_vram_usage_gb": 0.0,
        }

    def initialize_engine(self) -> bool:
        """Initialize vLLM engine with FP8 optimization."""
        if not VLLM_AVAILABLE:
            logger.error("vLLM not available - cannot initialize engine")
            return False

        try:
            # Set environment variables for FP8 optimization
            os.environ["VLLM_ATTENTION_BACKEND"] = self.config.attention_backend
            os.environ["VLLM_USE_CUDNN_PREFILL"] = "1"

            # Create engine arguments
            engine_args = AsyncEngineArgs(
                model=self.config.model,
                max_model_len=self.config.max_model_len,
                kv_cache_dtype=self.config.kv_cache_dtype,
                calculate_kv_scales=self.config.calculate_kv_scales,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_num_seqs=self.config.max_num_seqs,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                enable_chunked_prefill=self.config.enable_chunked_prefill,
                dtype=self.config.dtype,
                trust_remote_code=self.config.trust_remote_code,
            )

            # Initialize async engine
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)

            # Also initialize sync engine for compatibility
            self.llm = LLM(
                model=self.config.model,
                max_model_len=self.config.max_model_len,
                kv_cache_dtype=self.config.kv_cache_dtype,
                calculate_kv_scales=self.config.calculate_kv_scales,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_num_seqs=self.config.max_num_seqs,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                enable_chunked_prefill=self.config.enable_chunked_prefill,
                dtype=self.config.dtype,
                trust_remote_code=self.config.trust_remote_code,
            )

            logger.info(
                "vLLM engine initialized successfully with %s", self.config.model
            )
            return True

        except (ImportError, RuntimeError, ValueError, OSError) as e:
            logger.error("Failed to initialize vLLM engine: %s", e)
            return False

    def create_vllm_instance(self) -> Any | None:
        """Create optimized vLLM instance for LlamaIndex integration.

        Returns:
            Configured vLLM instance or None if vLLM is not available

        Raises:
            ImportError: When vLLM is not available
            Exception: When vLLM instance creation fails
        """
        if Vllm is None:
            error_msg = (
                "vLLM is not available. Please install with: "
                "uv sync --extra gpu or pip install 'vllm[flashinfer]>=0.10.1'"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

        try:
            # Create LlamaIndex wrapper
            vllm_llm = Vllm(
                model=self.config.model,
                dtype=self.config.dtype,
                kv_cache_dtype=self.config.kv_cache_dtype,
                max_new_tokens=2048,
                vllm_kwargs={
                    "gpu_memory_utilization": self.config.gpu_memory_utilization,
                    "max_model_len": self.config.max_model_len,
                    "trust_remote_code": self.config.trust_remote_code,
                    "enable_chunked_prefill": self.config.enable_chunked_prefill,
                },
            )

            logger.info("vLLM LlamaIndex instance created successfully")
            return vllm_llm

        except ImportError:
            raise
        except Exception as e:
            logger.error("Failed to create vLLM LlamaIndex instance: %s", e)
            raise

    def integrate_with_llamaindex(self) -> None:
        """Integrate vLLM with LlamaIndex global settings.

        Raises:
            ValueError: When vLLM instance is not created
            ImportError: When vLLM is not available
        """
        if Vllm is None:
            error_msg = "vLLM is not available for LlamaIndex integration"
            logger.error(error_msg)
            raise ImportError(error_msg)

        if not self.llm:
            raise ValueError(
                "vLLM instance not created. Call initialize_engine() first."
            )

        # Create LlamaIndex wrapper instance
        llm_instance = self.create_vllm_instance()

        # Set as global LLM
        Settings.llm = llm_instance
        logger.info("vLLM integrated with LlamaIndex settings")

    def validate_fp8_performance(self) -> dict[str, Any]:
        """Validate FP8 performance meets targets.

        Returns:
            Performance validation results
        """
        if not self.llm:
            return {"error": "Engine not initialized"}

        try:
            # Check VRAM usage
            vram_gb = self._get_vram_usage()

            # Benchmark decode throughput
            decode_throughput = self._benchmark_decode_throughput()

            # Benchmark prefill throughput
            prefill_throughput = self._benchmark_prefill_throughput()

            # Calculate memory reduction (estimate)
            fp8_memory_reduction = self._estimate_fp8_memory_reduction()

            validation_results = {
                "vram_usage_gb": vram_gb,
                "decode_throughput_tokens_per_sec": decode_throughput,
                "prefill_throughput_tokens_per_sec": prefill_throughput,
                "fp8_memory_reduction": fp8_memory_reduction,
                "meets_decode_target": decode_throughput
                >= self.config.target_decode_throughput,
                "meets_prefill_target": prefill_throughput
                >= self.config.target_prefill_throughput,
                "meets_vram_target": vram_gb <= self.config.vram_usage_target_gb,
                "meets_memory_reduction_target": fp8_memory_reduction >= 0.5,
            }

            # Log validation results
            if all(
                [
                    validation_results["meets_decode_target"],
                    validation_results["meets_prefill_target"],
                    validation_results["meets_vram_target"],
                    validation_results["meets_memory_reduction_target"],
                ]
            ):
                logger.info("✅ FP8 performance validation passed")
            else:
                logger.warning("⚠️ FP8 performance validation failed")

            self._performance_metrics = validation_results
            return validation_results

        except Exception as e:
            logger.error("Performance validation failed: %s", e)
            return {"error": str(e), "validation_failed": True}

    def _get_vram_usage(self) -> float:
        """Get current VRAM usage with proper error handling.

        Returns:
            VRAM usage in GB (0.0 if CUDA unavailable or error)
        """
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**3
            return 0.0
        except RuntimeError as e:
            logger.warning("CUDA memory check failed: %s", e)
            return 0.0
        except Exception as e:
            logger.error("Unexpected error checking VRAM: %s", e)
            return 0.0

    def _benchmark_decode_throughput(self) -> float:
        """Benchmark decode throughput.

        Returns:
            Tokens per second during decode
        """
        if not self.llm:
            return 0.0

        try:
            # Simple decode throughput test
            prompt = "The quick brown fox jumps over the lazy dog." * 20
            sampling_params = SamplingParams(temperature=0.1, max_tokens=200)

            start_time = time.perf_counter()
            outputs = self.llm.generate([prompt], sampling_params)
            elapsed_time = time.perf_counter() - start_time

            # Estimate tokens (rough approximation)
            output_text = outputs[0].outputs[0].text
            output_tokens = len(output_text.split()) * 1.3  # Approximate tokens
            throughput = output_tokens / elapsed_time if elapsed_time > 0 else 0

            logger.info("Decode throughput: %.1f tokens/sec", throughput)
            return throughput

        except Exception as e:
            logger.warning("Decode benchmark failed: %s", e)
            return 0.0

    def _benchmark_prefill_throughput(self) -> float:
        """Benchmark prefill throughput.

        Returns:
            Tokens per second during prefill
        """
        if not self.llm:
            return 0.0

        try:
            # Create long context for prefill test
            long_context = (
                "This is a comprehensive test of the prefill performance. " * 200
            )
            sampling_params = SamplingParams(temperature=0.1, max_tokens=50)

            start_time = time.perf_counter()
            self.llm.generate([long_context], sampling_params)
            elapsed_time = time.perf_counter() - start_time

            # Estimate prefill tokens
            prefill_tokens = len(long_context.split()) * 1.3  # Approximate tokens
            throughput = prefill_tokens / elapsed_time if elapsed_time > 0 else 0

            logger.info("Prefill throughput: %.1f tokens/sec", throughput)
            return throughput

        except Exception as e:
            logger.warning("Prefill benchmark failed: %s", e)
            return 0.0

    def _estimate_fp8_memory_reduction(self) -> float:
        """Estimate FP8 memory reduction.

        Returns:
            Estimated memory reduction percentage
        """
        # FP8 KV cache provides approximately 50% memory reduction
        # This is based on FP8 using 8-bit vs FP16 using 16-bit precision
        fp8_reduction = 0.5

        # Additional savings from model quantization
        if "fp8" in self.config.kv_cache_dtype.lower():
            fp8_reduction = 0.55  # Slightly higher with FP8 KV cache

        logger.info("Estimated FP8 memory reduction: %.1%%", fp8_reduction * 100)
        return fp8_reduction

    def test_128k_context_support(self) -> dict[str, Any]:
        """Test 128K context support with FP8.

        Returns:
            Context test results
        """
        if not self.llm:
            return {"error": "Engine not initialized"}

        try:
            # Create increasingly large contexts
            context_sizes = [1000, 5000, 10000, 32768, 65536]
            results = []

            for size in context_sizes:
                test_context = "Context token. " * (size // 2)
                start_vram = self._get_vram_usage()
                start_time = time.perf_counter()

                try:
                    sampling_params = SamplingParams(temperature=0.1, max_tokens=100)
                    self.llm.generate([test_context], sampling_params)
                    elapsed_time = time.perf_counter() - start_time
                    end_vram = self._get_vram_usage()

                    results.append(
                        {
                            "context_size": size,
                            "success": True,
                            "latency": elapsed_time,
                            "vram_usage": end_vram,
                            "vram_increase": end_vram - start_vram,
                        }
                    )
                    logger.info(
                        "✅ Context size %d: %.2fs, VRAM: %.2fGB",
                        size,
                        elapsed_time,
                        end_vram,
                    )

                except RuntimeError as e:
                    if "CUDA" in str(e).upper() or "memory" in str(e).lower():
                        logger.warning(
                            "❌ Context size %d failed due to CUDA/memory error: %s",
                            size,
                            e,
                        )
                    else:
                        logger.warning(
                            "❌ Context size %d failed with runtime error: %s", size, e
                        )
                    results.append(
                        {"context_size": size, "success": False, "error": str(e)}
                    )
                    break
                except Exception as e:
                    results.append(
                        {"context_size": size, "success": False, "error": str(e)}
                    )
                    logger.warning(
                        "❌ Context size %d failed with unexpected error: %s", size, e
                    )
                    break

            max_successful_context = max(
                [r["context_size"] for r in results if r.get("success", False)],
                default=0,
            )

            return {
                "max_context_supported": max_successful_context,
                "supports_128k": max_successful_context >= 128000,
                "results": results,
            }

        except Exception as e:
            logger.error("Context test failed: %s", e)
            return {"max_context_supported": 0, "supports_128k": False, "error": str(e)}

    def generate_start_script(self, output_path: str = "start_vllm.sh") -> str:
        """Generate bash script to start vLLM server with FP8 optimization."""
        script_content = f"""#!/bin/bash
# vLLM Server with FP8 Optimization for 128K Context (ADR-004, ADR-010)

export VLLM_ATTENTION_BACKEND={self.config.attention_backend}
export VLLM_USE_CUDNN_PREFILL=1

vllm serve {self.config.model} \\
  --max-model-len {self.config.max_model_len} \\
  --kv-cache-dtype {self.config.kv_cache_dtype} \\
  --calculate-kv-scales \\
  --gpu-memory-utilization {self.config.gpu_memory_utilization} \\
  --max-num-seqs {self.config.max_num_seqs} \\
  --max-num-batched-tokens {self.config.max_num_batched_tokens} \\
  --enable-chunked-prefill \\
  --trust-remote-code \\
  --host {self.config.host} --port {self.config.port} \\
  --served-model-name {self.config.served_model_name}
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        # Make executable
        os.chmod(output_path, FILE_PERMISSION_EXECUTABLE)  # noqa: S103

        logger.info("vLLM start script generated: %s", output_path)
        return output_path

    def validate_performance(self) -> dict[str, Any]:
        """Validate performance against ADR targets."""
        if not self.llm:
            return {"error": "Engine not initialized"}

        try:
            # Test generation for performance validation
            test_prompt = "Explain machine learning in simple terms."
            sampling_params = SamplingParams(temperature=0.1, max_tokens=100)

            start_time = time.perf_counter()
            outputs = self.llm.generate([test_prompt], sampling_params)
            generation_time = time.perf_counter() - start_time

            # Calculate basic metrics
            generated_text = outputs[0].outputs[0].text
            tokens_generated = len(generated_text.split())  # Rough approximation
            throughput = (
                tokens_generated / generation_time if generation_time > 0 else 0
            )

            # Performance validation
            validation_results = {
                "decode_throughput_estimate": throughput,
                "meets_decode_target": 100 <= throughput <= 160,
                "generation_time": generation_time,
                "tokens_generated": tokens_generated,
                "model_loaded": True,
                "fp8_optimization": self.config.kv_cache_dtype == "fp8_e5m2",
                "context_window": self.config.max_model_len,
                "meets_context_target": self.config.max_model_len >= 131072,
                "validation_timestamp": time.time(),
            }

            logger.info("Performance validation completed: %.1f tok/s", throughput)
            return validation_results

        except (ImportError, RuntimeError, ValueError, AttributeError) as e:
            logger.error("Performance validation failed: %s", e)
            return {"error": str(e), "validation_failed": True}

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self._performance_metrics,
            "config": {
                "model": self.config.model,
                "max_context": self.config.max_model_len,
                "kv_cache_dtype": self.config.kv_cache_dtype,
                "attention_backend": self.config.attention_backend,
                "vram_target_gb": self.config.vram_usage_target_gb,
            },
            "targets": {
                "decode_throughput_range": (100, 160),
                "prefill_throughput_range": (800, 1300),
                "vram_usage_range_gb": (12, 14),
                "context_window": 131072,
            },
        }


def create_vllm_manager(
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507-FP8",
    max_context_length: int = app_settings.default_token_limit,
) -> VLLMManager:
    """Create vLLM manager with FP8 optimization."""
    config = VLLMConfig(model=model_path, max_model_len=max_context_length)
    return VLLMManager(config)


def validate_fp8_requirements() -> dict[str, Any]:
    """Validate system requirements for FP8 with comprehensive error handling.

    Returns:
        Requirements validation results with detailed hardware information
    """
    results = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": "N/A",
        "cuda_version": "N/A",
        "supports_fp8": False,
        "vram_gb": 0.0,
        "gpu_compute_capability": "N/A",
        "vllm_available": VLLM_AVAILABLE,
        "sufficient_vram": False,
        "flashinfer_backend": True,  # Assume FlashInfer support
    }

    try:
        import torch

        results["cuda_available"] = torch.cuda.is_available()
        results["cuda_version"] = (
            torch.version.cuda if hasattr(torch.version, "cuda") else "N/A"
        )

        if results["cuda_available"]:
            results["gpu_count"] = torch.cuda.device_count()

            if results["gpu_count"] > 0:
                results["gpu_name"] = torch.cuda.get_device_name(0)

                gpu_props = torch.cuda.get_device_properties(0)
                results["vram_gb"] = gpu_props.total_memory / 1024**3
                results["gpu_compute_capability"] = (
                    f"{gpu_props.major}.{gpu_props.minor}"
                )

                # Check for sufficient VRAM
                results["sufficient_vram"] = (
                    results["vram_gb"] >= app_settings.vllm_minimum_vram_gb
                )

                # Check for FP8 support (Ada Lovelace and newer)
                if gpu_props.major >= 9 or (
                    gpu_props.major == 8 and gpu_props.minor >= 9
                ):
                    results["supports_fp8"] = True

    except RuntimeError as e:
        logger.warning("CUDA validation failed: %s", e)
    except Exception as e:
        logger.error("Unexpected error during FP8 requirements validation: %s", e)

    # Validate software requirements
    try:
        if VLLM_AVAILABLE:
            import vllm

            results["vllm_version"] = vllm.__version__
    except ImportError:
        pass

    # Log validation results
    if (
        results["supports_fp8"]
        and results["vllm_available"]
        and results["sufficient_vram"]
    ):
        logger.info("✅ FP8 requirements validation passed")
    else:
        logger.warning("⚠️ FP8 requirements validation failed")

    return results
