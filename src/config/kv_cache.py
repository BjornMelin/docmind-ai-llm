"""FP8 KV Cache optimization and context management configuration.

This module implements ADR-009 compliant FP8 KV cache optimization for vLLM
with 50% memory reduction, 128K context window support, and intelligent
context management for document processing pipeline.

Key Features:
- FP8 E5M2 KV cache quantization for 50% VRAM reduction
- 128K context window with intelligent trimming
- vLLM FlashInfer backend optimization
- Dynamic context management for large documents
- Memory pressure monitoring and automatic scaling
- Performance targets: 120-180 tok/s decode, 900-1400 tok/s prefill
"""

import gc
import time
from typing import Any

import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import app_settings


class FP8CacheConfig(BaseModel):
    """FP8 KV cache configuration parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # FP8 KV Cache Settings
    kv_cache_dtype: str = Field(
        default="fp8_e5m2",
        description="FP8 KV cache dtype (fp8_e5m2 for 50% memory reduction)",
    )
    calculate_kv_scales: bool = Field(
        default=True, description="Required for FP8 KV cache accuracy"
    )

    # vLLM FlashInfer Backend
    attention_backend: str = Field(
        default="FLASHINFER",
        description="vLLM attention backend (FLASHINFER for optimal performance)",
    )
    use_cudnn_prefill: bool = Field(
        default=True, description="Use CUDNN for prefill optimization"
    )
    enable_chunked_prefill: bool = Field(
        default=True, description="Enable chunked prefill for large contexts"
    )

    # Context Window Management
    max_model_len: int = Field(
        default=131072,
        ge=8192,
        le=200000,
        description="Maximum model context length (128K + buffer)",
    )
    max_token_limit: int = Field(
        default=120000,
        ge=10000,
        le=180000,
        description="Maximum tokens before intelligent trimming",
    )
    context_buffer_size: int = Field(
        default=8192, ge=1024, le=16384, description="Buffer size for context overflow"
    )

    # Memory Management
    gpu_memory_utilization: float = Field(
        default=0.85,
        ge=0.5,
        le=0.95,
        description="GPU memory utilization (85% for RTX 4090)",
    )
    kv_cache_memory_per_token: int = Field(
        default=1024,
        ge=512,
        le=2048,
        description="Memory per token with FP8 KV cache (bytes)",
    )
    enable_memory_monitoring: bool = Field(
        default=True, description="Enable dynamic memory monitoring"
    )

    # Performance Optimization
    max_num_batched_tokens: int = Field(
        default=8192,
        ge=1024,
        le=16384,
        description="Maximum batched tokens for performance",
    )
    max_num_seqs: int = Field(
        default=16, ge=1, le=64, description="Maximum concurrent sequences"
    )
    prefill_chunk_size: int = Field(
        default=4096, ge=1024, le=8192, description="Chunk size for prefill operations"
    )


class ContextWindow(BaseModel):
    """Context window state and management."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_tokens: int = Field(default=0, description="Current token count")
    available_tokens: int = Field(description="Available tokens before limit")
    total_capacity: int = Field(description="Total context capacity")
    buffer_size: int = Field(description="Buffer size for overflow")
    trimming_threshold: float = Field(
        default=0.9, description="Threshold for context trimming"
    )

    @property
    def utilization(self) -> float:
        """Current context utilization ratio (0.0-1.0)."""
        return (
            self.current_tokens / self.total_capacity
            if self.total_capacity > 0
            else 0.0
        )

    @property
    def needs_trimming(self) -> bool:
        """Whether context needs trimming."""
        return self.utilization >= self.trimming_threshold


class MemoryStats(BaseModel):
    """Memory usage statistics and monitoring."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_vram_gb: float = Field(description="Total VRAM in GB")
    used_vram_gb: float = Field(description="Used VRAM in GB")
    kv_cache_vram_gb: float = Field(default=0.0, description="KV cache VRAM usage")
    utilization_ratio: float = Field(description="VRAM utilization (0.0-1.0)")
    memory_pressure: bool = Field(
        default=False, description="High memory pressure detected"
    )
    fp8_savings_gb: float = Field(
        default=0.0, description="Memory saved with FP8 quantization"
    )

    @property
    def available_vram_gb(self) -> float:
        """Available VRAM in GB."""
        return self.total_vram_gb - self.used_vram_gb

    @property
    def is_under_pressure(self) -> bool:
        """Whether system is under memory pressure."""
        return self.utilization_ratio > 0.9 or self.memory_pressure


class KVCacheManager:
    """FP8 KV cache manager with context optimization and memory monitoring.

    This manager implements ADR-009 requirements for FP8 KV cache optimization
    with intelligent context management for document processing:

    - FP8 E5M2 quantization for 50% memory reduction
    - 128K context window with intelligent trimming
    - vLLM FlashInfer backend optimization
    - Dynamic memory monitoring and scaling
    - Context buffer management for large documents
    """

    def __init__(self, settings: Any | None = None):
        """Initialize KVCacheManager.

        Args:
            settings: DocMind configuration settings. Uses app_settings if None.
        """
        self.settings = settings or app_settings

        # Load FP8 cache configuration
        self.config = self._load_fp8_config()

        # Context window management
        self.context_window = ContextWindow(
            current_tokens=0,
            available_tokens=self.config.max_token_limit,
            total_capacity=self.config.max_model_len,
            buffer_size=self.config.context_buffer_size,
            trimming_threshold=0.9,
        )

        # Memory monitoring
        self._memory_stats: MemoryStats | None = None
        self._last_memory_check = 0.0
        self._memory_check_interval = 5.0  # seconds

        # Performance tracking
        self._token_processing_times: list[float] = []
        self._context_trims = 0
        self._memory_pressure_events = 0

        logger.info(
            "KVCacheManager initialized: FP8 cache={}, context_limit={}, "
            "memory_util={}",
            self.config.kv_cache_dtype,
            self.config.max_token_limit,
            self.config.gpu_memory_utilization,
        )

    def _load_fp8_config(self) -> FP8CacheConfig:
        """Load FP8 cache configuration from settings.

        Returns:
            FP8CacheConfig with optimized parameters
        """
        return FP8CacheConfig(
            # FP8 KV cache from settings
            kv_cache_dtype=getattr(self.settings, "vllm_kv_cache_dtype", "fp8_e5m2"),
            calculate_kv_scales=getattr(
                self.settings, "vllm_calculate_kv_scales", True
            ),
            # vLLM backend from settings
            attention_backend=getattr(
                self.settings, "vllm_attention_backend", "FLASHINFER"
            ),
            use_cudnn_prefill=getattr(self.settings, "vllm_use_cudnn_prefill", True),
            enable_chunked_prefill=getattr(
                self.settings, "vllm_enable_chunked_prefill", True
            ),
            # Context management
            max_model_len=getattr(self.settings, "default_token_limit", 131072),
            max_token_limit=getattr(self.settings, "vllm_max_token_limit", 120000),
            context_buffer_size=getattr(self.settings, "context_buffer_size", 8192),
            # Memory settings
            gpu_memory_utilization=getattr(
                self.settings, "vllm_gpu_memory_utilization", 0.85
            ),
            kv_cache_memory_per_token=getattr(
                self.settings, "vllm_kv_cache_memory_per_token", 1024
            ),
            # Performance settings
            max_num_batched_tokens=getattr(
                self.settings, "vllm_max_num_batched_tokens", 8192
            ),
            max_num_seqs=getattr(self.settings, "vllm_max_num_seqs", 16),
        )

    def get_vllm_engine_args(self) -> dict[str, Any]:
        """Get vLLM engine arguments with FP8 optimization.

        Returns:
            Dictionary of vLLM engine arguments
        """
        vllm_args = {
            # Model and tokenizer
            "model": getattr(
                self.settings, "model_name", "Qwen/Qwen3-4B-Instruct-2507-FP8"
            ),
            "tokenizer": getattr(
                self.settings, "model_name", "Qwen/Qwen3-4B-Instruct-2507-FP8"
            ),
            # FP8 KV Cache optimization
            "kv_cache_dtype": self.config.kv_cache_dtype,
            "quantization_param_path": None,  # Auto-detect for FP8
            # Context and memory
            "max_model_len": self.config.max_model_len,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            # FlashInfer backend
            "attention_backend": self.config.attention_backend,
            "use_v2_block_manager": True,  # Required for FlashInfer
            # Performance optimization
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            # Advanced settings
            "enforce_eager": False,
            "disable_log_stats": False,
            "disable_log_requests": True,
        }

        # Add FP8-specific settings if available
        if hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            vllm_args.update(
                {
                    "dtype": "auto",  # Auto-detect FP8 support
                    "load_format": "auto",
                    "device": "cuda",
                }
            )

        return vllm_args

    def update_context_usage(self, token_count: int) -> None:
        """Update context window usage.

        Args:
            token_count: Current token count
        """
        self.context_window.current_tokens = token_count
        self.context_window.available_tokens = max(
            0, self.config.max_token_limit - token_count
        )

        if self.context_window.needs_trimming:
            logger.warning(
                f"Context approaching limit: {token_count}/"
                f"{self.config.max_token_limit} "
                f"({self.context_window.utilization:.1%} utilization)"
            )

    def calculate_trim_amount(self, target_utilization: float = 0.7) -> int:
        """Calculate how many tokens to trim for target utilization.

        Args:
            target_utilization: Target context utilization (0.0-1.0)

        Returns:
            Number of tokens to trim
        """
        target_tokens = int(self.config.max_token_limit * target_utilization)
        current_tokens = self.context_window.current_tokens

        if current_tokens <= target_tokens:
            return 0

        # Trim to target plus buffer
        trim_amount = current_tokens - target_tokens + self.config.context_buffer_size

        logger.info(
            f"Context trimming: removing {trim_amount} tokens "
            f"({current_tokens} -> {target_tokens})"
        )

        self._context_trims += 1
        return trim_amount

    def should_enable_chunked_prefill(self, input_length: int) -> bool:
        """Determine if chunked prefill should be enabled.

        Args:
            input_length: Input sequence length

        Returns:
            True if chunked prefill should be used
        """
        # Enable chunked prefill for large inputs
        return (
            input_length > self.config.prefill_chunk_size
            and self.config.enable_chunked_prefill
        )

    def get_optimal_batch_size(self, sequence_length: int) -> int:
        """Get optimal batch size based on sequence length and memory.

        Args:
            sequence_length: Average sequence length

        Returns:
            Optimal batch size
        """
        # Update memory stats if needed
        self._maybe_update_memory_stats()

        base_batch_size = self.config.max_num_seqs

        # Reduce batch size for long sequences
        if sequence_length > 4096:
            base_batch_size = max(1, base_batch_size // 2)

        # Reduce batch size under memory pressure
        if self._memory_stats and self._memory_stats.is_under_pressure:
            base_batch_size = max(1, base_batch_size // 2)
            self._memory_pressure_events += 1

        return base_batch_size

    def _maybe_update_memory_stats(self) -> None:
        """Update memory statistics if interval has elapsed."""
        current_time = time.time()
        if current_time - self._last_memory_check < self._memory_check_interval:
            return

        self._last_memory_check = current_time

        if not torch.cuda.is_available():
            return

        try:
            # Get GPU memory stats
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)

            total_gb = total_memory / (1024**3)
            used_gb = allocated_memory / (1024**3)
            utilization = used_gb / total_gb

            # Estimate KV cache memory (rough approximation)
            kv_cache_gb = (
                self.context_window.current_tokens
                * self.config.kv_cache_memory_per_token
                / (1024**3)
            )

            # Estimate FP8 savings (50% reduction vs FP16)
            fp8_savings_gb = kv_cache_gb  # Savings = original FP16 size - FP8 size

            self._memory_stats = MemoryStats(
                total_vram_gb=total_gb,
                used_vram_gb=used_gb,
                kv_cache_vram_gb=kv_cache_gb,
                utilization_ratio=utilization,
                memory_pressure=utilization > 0.9,
                fp8_savings_gb=fp8_savings_gb,
            )

        except Exception as e:
            logger.debug(f"Could not update memory stats: {str(e)}")

    def get_memory_stats(self) -> MemoryStats | None:
        """Get current memory statistics.

        Returns:
            MemoryStats if available, None otherwise
        """
        self._maybe_update_memory_stats()
        return self._memory_stats

    def optimize_for_document_processing(self) -> dict[str, Any]:
        """Get optimized configuration for document processing workload.

        Returns:
            Optimized vLLM configuration
        """
        base_args = self.get_vllm_engine_args()

        # Document processing optimizations
        doc_optimizations = {
            # Optimize for longer sequences (document chunks)
            "max_num_seqs": min(8, self.config.max_num_seqs),  # Fewer concurrent seqs
            "max_num_batched_tokens": max(
                4096, self.config.max_num_batched_tokens // 2
            ),
            # Enable aggressive KV caching for document reuse
            "enable_prefix_caching": True,
            "max_num_blocks_per_seq": None,  # Auto-calculate
            # Memory optimization for long documents
            "swap_space": 4,  # 4GB swap space for large documents
            "cpu_offload_gb": 0,  # Keep everything on GPU with FP8
        }

        base_args.update(doc_optimizations)
        return base_args

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        avg_token_time = (
            sum(self._token_processing_times) / len(self._token_processing_times)
            if self._token_processing_times
            else 0.0
        )

        return {
            "context_utilization": self.context_window.utilization,
            "available_tokens": self.context_window.available_tokens,
            "context_trims": self._context_trims,
            "memory_pressure_events": self._memory_pressure_events,
            "avg_token_processing_time_ms": avg_token_time * 1000,
            "total_token_operations": len(self._token_processing_times),
            "fp8_cache_enabled": self.config.kv_cache_dtype == "fp8_e5m2",
            "flashinfer_enabled": self.config.attention_backend == "FLASHINFER",
        }

    def track_token_processing(self, processing_time: float) -> None:
        """Track token processing time for performance monitoring.

        Args:
            processing_time: Processing time in seconds
        """
        self._token_processing_times.append(processing_time)

        # Keep only recent samples (sliding window)
        if len(self._token_processing_times) > 1000:
            self._token_processing_times = self._token_processing_times[-500:]

    def estimate_memory_usage(self, token_count: int) -> dict[str, float]:
        """Estimate memory usage for given token count.

        Args:
            token_count: Number of tokens

        Returns:
            Dictionary with memory usage estimates
        """
        # Base model memory (Qwen3-4B-FP8 ≈ 4GB)
        base_model_gb = 4.0

        # KV cache memory with FP8 optimization
        kv_cache_bytes = token_count * self.config.kv_cache_memory_per_token
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # Total estimated usage
        total_gb = base_model_gb + kv_cache_gb

        # FP8 savings (50% reduction vs FP16)
        fp16_kv_cache_gb = kv_cache_gb * 2  # FP16 would be 2x larger
        savings_gb = fp16_kv_cache_gb - kv_cache_gb

        return {
            "base_model_gb": base_model_gb,
            "kv_cache_gb": kv_cache_gb,
            "total_usage_gb": total_gb,
            "fp8_savings_gb": savings_gb,
            "savings_percentage": (savings_gb / fp16_kv_cache_gb) * 100
            if fp16_kv_cache_gb > 0
            else 0.0,
        }

    def cleanup_memory(self) -> None:
        """Clean up GPU memory and reset statistics."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        # Reset context window
        self.context_window.current_tokens = 0
        self.context_window.available_tokens = self.config.max_token_limit

        logger.info("Memory cleanup completed")


# Factory function for easy instantiation
def create_kv_cache_manager(settings: Any | None = None) -> KVCacheManager:
    """Factory function to create KVCacheManager instance.

    Args:
        settings: Optional DocMind settings. Uses app_settings if None.

    Returns:
        Configured KVCacheManager instance
    """
    return KVCacheManager(settings)


# Convenience function for vLLM engine creation
def get_optimized_vllm_config(settings: Any | None = None) -> dict[str, Any]:
    """Get optimized vLLM configuration with FP8 KV cache.

    Args:
        settings: Optional DocMind settings. Uses app_settings if None.

    Returns:
        Dictionary with vLLM engine configuration
    """
    manager = create_kv_cache_manager(settings)
    return manager.optimize_for_document_processing()
