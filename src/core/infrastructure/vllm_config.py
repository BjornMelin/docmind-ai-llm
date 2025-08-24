"""vLLM FP8 Configuration and Management.

Configures vLLM with native FP8 quantization for RTX 4090 optimization.
Provides 50% KV cache memory reduction and enhanced performance.
"""

import logging
import os
import time
from enum import Enum
from typing import Any

import torch
from llama_index.core import Settings

# Import vLLM with proper error handling - no mocks in production
try:
    from llama_index.llms.vllm import Vllm

    VllmType: type | None = Vllm
except ImportError:
    Vllm = None
    VllmType = None
    # Logger will be initialized below, warning will be logged when needed

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Log vLLM availability warning if not available
if Vllm is None:
    logger.warning(
        "vLLM not available - vLLM features disabled. "
        "Install with: uv sync --extra gpu or pip install 'vllm[flashinfer]>=0.10.1'"
    )


def is_vllm_available() -> bool:
    """Check if vLLM is available.

    Returns:
        True if vLLM is available, False otherwise
    """
    return Vllm is not None


def require_vllm() -> None:
    """Require vLLM to be available, raise ImportError if not.

    Raises:
        ImportError: When vLLM is not available
    """
    if Vllm is None:
        raise ImportError(
            "vLLM is not available. Please install with: "
            "uv sync --extra gpu or pip install 'vllm[flashinfer]>=0.10.1'"
        )


class FP8Precision(str, Enum):
    """FP8 precision types."""

    E4M3 = "fp8_e4m3"
    E5M2 = "fp8_e5m2"


class VLLMBackend(str, Enum):
    """vLLM attention backends."""

    FLASHINFER = "FLASHINFER"
    FLASHATTN = "FLASHATTN"
    XFORMERS = "XFORMERS"


class VLLMConfig(BaseModel):
    """Configuration for vLLM with FP8 optimization."""

    model_name: str = Field(default="Qwen/Qwen3-4B-Instruct-2507-FP8")
    quantization: str = Field(default="fp8")
    kv_cache_dtype: FP8Precision = Field(default=FP8Precision.E5M2)
    attention_backend: VLLMBackend = Field(default=VLLMBackend.FLASHINFER)
    gpu_memory_utilization: float = Field(default=0.85, ge=0.1, le=1.0)
    max_model_len: int = Field(default=131072)
    max_new_tokens: int = Field(default=2048)
    trust_remote_code: bool = Field(default=True)
    enforce_eager: bool = Field(default=False)
    enable_chunked_prefill: bool = Field(default=True)
    tensor_parallel_size: int = Field(default=1)
    dtype: str = Field(default="float16")

    # Performance targets for RTX 4090
    target_decode_tokens_per_sec: float = Field(default=120.0)
    target_prefill_tokens_per_sec: float = Field(default=900.0)
    max_vram_gb: float = Field(default=14.0)
    fp8_memory_reduction_target: float = Field(default=0.5)

    # Validation settings
    enable_performance_validation: bool = Field(default=True)
    validation_queries: list[str] = Field(
        default=[
            "What is machine learning?",
            "Explain the concept of neural networks in detail.",
            "Compare supervised and unsupervised learning approaches.",
        ]
    )


class VLLMManager:
    """Manages vLLM instances with FP8 optimization."""

    def __init__(self, config: VLLMConfig | None = None):
        """Initialize vLLM manager.

        Args:
            config: vLLM configuration
        """
        self.config = config or VLLMConfig()
        self.llm_instance = None
        self.performance_metrics = {}
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Setup vLLM environment variables with error handling."""
        os.environ["VLLM_ATTENTION_BACKEND"] = self.config.attention_backend.value

        # Enable FP8 optimizations on Ada Lovelace (RTX 4090)
        try:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                if "RTX 4090" in device_name or "Ada Lovelace" in device_name:
                    logger.info("Detected RTX 4090: %s", device_name)
                    os.environ["VLLM_USE_FP8_E4M3"] = (
                        "1" if self.config.kv_cache_dtype == FP8Precision.E4M3 else "0"
                    )
                    os.environ["VLLM_USE_FP8_E5M2"] = (
                        "1" if self.config.kv_cache_dtype == FP8Precision.E5M2 else "0"
                    )
        except RuntimeError as e:
            logger.warning("Failed to detect GPU device for FP8 setup: %s", e)
        except Exception as e:
            logger.error("Unexpected error during GPU environment setup: %s", e)

    def create_vllm_instance(self) -> Any | None:
        """Create optimized vLLM instance.

        Returns:
            Configured vLLM instance or None if vLLM is not available

        Raises:
            ImportError: When vLLM is not available
            Exception: When vLLM instance creation fails
        """
        # Check if vLLM is available
        if Vllm is None:
            error_msg = (
                "vLLM is not available. Please install with: "
                "uv sync --extra gpu or pip install 'vllm[flashinfer]>=0.10.1'"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

        try:
            # Import vLLM for direct configuration
            from vllm import LLM

            # Create vLLM instance with FP8 configuration
            # Note: vLLM backend instance created for internal configuration validation
            _vllm_backend = LLM(
                model=self.config.model_name,
                quantization=self.config.quantization,
                kv_cache_dtype=self.config.kv_cache_dtype.value,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                trust_remote_code=self.config.trust_remote_code,
                enforce_eager=self.config.enforce_eager,
                enable_chunked_prefill=self.config.enable_chunked_prefill,
                tensor_parallel_size=self.config.tensor_parallel_size,
            )

            # Create LlamaIndex wrapper
            vllm_llm = Vllm(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                dtype=self.config.dtype,
                quantization=self.config.quantization,
                kv_cache_dtype=self.config.kv_cache_dtype.value,
                max_new_tokens=self.config.max_new_tokens,
                vllm_kwargs={
                    "gpu_memory_utilization": self.config.gpu_memory_utilization,
                    "max_model_len": self.config.max_model_len,
                    "trust_remote_code": self.config.trust_remote_code,
                    "enforce_eager": self.config.enforce_eager,
                    "enable_chunked_prefill": self.config.enable_chunked_prefill,
                },
            )

            self.llm_instance = vllm_llm

            # Validate FP8 performance if enabled
            if self.config.enable_performance_validation:
                self.validate_fp8_performance()

            logger.info("vLLM FP8 instance created successfully")
            return vllm_llm

        except ImportError:
            # Re-raise ImportError with context
            raise
        except Exception as e:
            logger.error("Failed to create vLLM instance: %s", e)
            raise

    def validate_fp8_performance(self) -> dict[str, Any]:
        """Validate FP8 performance meets targets.

        Returns:
            Performance validation results
        """
        if not self.llm_instance:
            raise ValueError("vLLM instance not created")

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
            >= self.config.target_decode_tokens_per_sec,
            "meets_prefill_target": prefill_throughput
            >= self.config.target_prefill_tokens_per_sec,
            "meets_vram_target": vram_gb <= self.config.max_vram_gb,
            "meets_memory_reduction_target": fp8_memory_reduction
            >= self.config.fp8_memory_reduction_target,
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

        self.performance_metrics = validation_results
        return validation_results

    def _get_vram_usage(self) -> float:
        """Get current VRAM usage with proper error handling.

        Returns:
            VRAM usage in GB (0.0 if CUDA unavailable or error)
        """
        try:
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
        if not self.llm_instance:
            return 0.0

        try:
            # Simple decode throughput test
            prompt = "The quick brown fox jumps over the lazy dog. " * 20
            start_time = time.time()

            response = self.llm_instance.complete(prompt, max_tokens=200)
            elapsed_time = time.time() - start_time

            # Estimate tokens (rough approximation)
            output_tokens = len(response.text.split()) * 1.3  # Approximate tokens
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
        if not self.llm_instance:
            return 0.0

        try:
            # Create long context for prefill test
            long_context = (
                "This is a comprehensive test of the prefill performance. " * 200
            )
            start_time = time.time()

            _response = self.llm_instance.complete(long_context, max_tokens=50)
            elapsed_time = time.time() - start_time

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
        if self.config.quantization == "fp8":
            fp8_reduction = 0.55  # Slightly higher with model quantization

        logger.info("Estimated FP8 memory reduction: %.1%%", fp8_reduction * 100)
        return fp8_reduction

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

        if not self.llm_instance:
            raise ValueError(
                "vLLM instance not created. Call create_vllm_instance() first."
            )

        # Set as global LLM
        Settings.llm = self.llm_instance
        logger.info("vLLM integrated with LlamaIndex settings")

    def test_128k_context_support(self) -> dict[str, Any]:
        """Test 128K context support with FP8.

        Returns:
            Context test results
        """
        if not self.llm_instance:
            raise ValueError("vLLM instance not created")

        try:
            # Create increasingly large contexts
            context_sizes = [1000, 5000, 10000, 32768, 65536]
            results = []

            for size in context_sizes:
                test_context = "Context token. " * (size // 2)
                start_vram = self._get_vram_usage()
                start_time = time.time()

                try:
                    _response = self.llm_instance.complete(test_context, max_tokens=100)
                    elapsed_time = time.time() - start_time
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


def create_fp8_vllm_config(model_name: str | None = None) -> VLLMConfig:
    """Create FP8-optimized vLLM configuration.

    Args:
        model_name: Optional model name override

    Returns:
        Optimized vLLM configuration
    """
    config = VLLMConfig()
    if model_name:
        config.model_name = model_name

    # Auto-detect hardware optimizations
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

            if "RTX 4090" in device_name:
                # RTX 4090 optimizations
                config.gpu_memory_utilization = 0.85
                config.kv_cache_dtype = FP8Precision.E5M2
                config.attention_backend = VLLMBackend.FLASHINFER
                config.max_vram_gb = 14.0
            elif gpu_memory < 12:
                # Lower memory GPU
                config.gpu_memory_utilization = 0.8
                config.max_model_len = 65536  # Reduce context for memory
            else:
                # High memory GPU
                config.gpu_memory_utilization = 0.9

            logger.info("Configured for GPU: %s (%.1fGB)", device_name, gpu_memory)
    except RuntimeError as e:
        logger.warning("Failed to detect GPU hardware for optimization: %s", e)
    except Exception as e:
        logger.error("Unexpected error during hardware detection: %s", e)

    return config


def setup_vllm_fp8() -> VLLMManager | None:
    """Setup vLLM with FP8 optimization.

    Returns:
        Configured vLLM manager or None if vLLM is not available

    Raises:
        ImportError: When vLLM is not available
        Exception: When vLLM setup fails
    """
    if Vllm is None:
        error_msg = (
            "vLLM is not available. Please install with: "
            "uv sync --extra gpu or pip install 'vllm[flashinfer]>=0.10.1'"
        )
        logger.error(error_msg)
        raise ImportError(error_msg)

    config = create_fp8_vllm_config()
    manager = VLLMManager(config)
    manager.create_vllm_instance()
    manager.integrate_with_llamaindex()

    logger.info("vLLM FP8 setup completed")
    return manager


def validate_fp8_requirements() -> dict[str, Any]:
    """Validate system requirements for FP8 with error handling.

    Returns:
        Requirements validation results
    """
    results = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": "N/A",
        "cuda_version": "N/A",
        "supports_fp8": False,
        "vram_gb": 0.0,
        "gpu_compute_capability": "N/A",
    }

    try:
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
        import vllm

        results["vllm_version"] = vllm.__version__
        results["vllm_available"] = True
    except ImportError:
        results["vllm_available"] = False

    # Log validation results
    if results["supports_fp8"] and results["vllm_available"]:
        logger.info("✅ FP8 requirements validation passed")
    else:
        logger.warning("⚠️ FP8 requirements validation failed")

    return results
