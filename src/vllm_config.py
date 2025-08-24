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

from src.config.settings import settings

# Constants
MAX_TOKEN_LIMIT = 120000  # Trim at 120K (8K buffer for 128K limit)
KV_CACHE_MEMORY_PER_TOKEN = 1024  # bytes per token with FP8
KV_CACHE_GB_AT_128K = 8.0  # ~8GB KV cache at 128K
CHARS_PER_TOKEN_AVERAGE = 4  # Simplified estimation - 4 chars per token average
FILE_PERMISSION_EXECUTABLE = 0o755
MINIMUM_VRAM_REQUIREMENT_GB = 12


class VLLMConfig(BaseModel):
    """Configuration for vLLM backend serving FP8 model with FlashInfer optimization."""

    # Model configuration
    model: str = Field(
        default="Qwen/Qwen3-4B-Instruct-2507-FP8",
        description="FP8 quantized model path",
    )
    max_model_len: int = Field(
        default=settings.context_window_size,  # 128K context (hardware-constrained)
        description="Maximum context length in tokens",
    )
    gpu_memory_utilization: float = Field(
        default=0.95, description="GPU memory utilization ratio"
    )

    # FP8 KV Cache Optimization (ADR-004, ADR-010)
    kv_cache_dtype: str = Field(
        default="fp8_e5m2", description="FP8 KV cache for 50% memory reduction"
    )
    calculate_kv_scales: bool = Field(
        default=True, description="Required for FP8 KV cache"
    )
    attention_backend: str = Field(
        default="FLASHINFER", description="FlashInfer backend for FP8 acceleration"
    )
    enable_chunked_prefill: bool = Field(default=True)
    use_cudnn_prefill: bool = Field(default=True)

    # Memory optimization
    max_num_seqs: int = Field(default=1, description="Single sequence for 128K context")
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


class ContextManager:
    """Manages 128K context window with intelligent trimming strategies.

    Implements ADR-004 and ADR-011 for context management.
    """

    def __init__(self):
        """Initialize context manager with 128K context and FP8 KV cache settings."""
        self.max_context_tokens = (
            settings.context_window_size  # 128K context (hardware-constrained)
        )
        self.trim_threshold = MAX_TOKEN_LIMIT  # Trim at 120K (8K buffer for 128K limit)
        self.preserve_ratio = 0.3  # Keep 30% of oldest context for continuity

        # Memory calculations for FP8 KV cache (ADR-010)
        self.kv_cache_memory_per_token = (
            KV_CACHE_MEMORY_PER_TOKEN  # bytes per token with FP8
        )
        self.total_kv_cache_gb_at_128k = KV_CACHE_GB_AT_128K  # ~8GB KV cache at 128K

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
    max_context_length: int = settings.context_window_size,
) -> VLLMManager:
    """Create vLLM manager with FP8 optimization."""
    config = VLLMConfig(model=model_path, max_model_len=max_context_length)
    return VLLMManager(config)


def validate_fp8_requirements() -> dict[str, bool]:
    """Validate FP8 optimization requirements."""
    requirements = {
        "vllm_available": VLLM_AVAILABLE,
        "cuda_available": True,  # Assume CUDA is available
        "fp8_support": True,  # Assume FP8 support
        "flashinfer_backend": True,  # Assume FlashInfer support
    }

    try:
        import torch

        requirements["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            # Check GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024**3)
            requirements["sufficient_vram"] = memory_gb >= MINIMUM_VRAM_REQUIREMENT_GB
        else:
            requirements["sufficient_vram"] = False
    except ImportError:
        requirements["cuda_available"] = False
        requirements["sufficient_vram"] = False

    return requirements
