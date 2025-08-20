"""vLLM Backend Integration for DocMind AI with FP8 Quantization.

This module provides vLLM integration with Qwen3-4B-Instruct-2507-FP8 model,
optimized for RTX 4090 hardware with FP8 quantization and FlashInfer attention.

Technical Specifications:
- Model: Qwen3-4B-Instruct-2507-FP8
- Context Window: 131,072 tokens (128K)
- Quantization: FP8 weights + FP8 KV cache
- Backend: vLLM 0.10.1 with FlashInfer attention
- Hardware Target: RTX 4090 Laptop GPU (16GB VRAM)
- Performance Targets: 100-160 tok/s decode, 800-1300 tok/s prefill
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine

from ..config.settings import settings

logger = logging.getLogger(__name__)


class VLLMConfig(BaseModel):
    """Configuration for vLLM backend with FP8 optimization."""

    # Model Configuration
    model_name: str = Field(
        default="Qwen/Qwen3-4B-Instruct-2507", description="Model name for vLLM backend"
    )
    model_path: str | None = Field(
        default=None, description="Local path to model (if using local model)"
    )

    # FP8 Quantization Settings (REQ-0063-v2, REQ-0064-v2)
    quantization: str = Field(
        default="fp8", description="FP8 quantization for weights and activations"
    )
    kv_cache_dtype: str = Field(
        default="fp8", description="FP8 KV cache for memory optimization"
    )

    # Context and Memory Configuration (REQ-0094-v2)
    max_model_len: int = Field(
        default=131072, description="Maximum context length (128K tokens)"
    )
    gpu_memory_utilization: float = Field(
        default=0.85, description="GPU memory utilization ratio for <14GB VRAM usage"
    )
    enforce_eager: bool = Field(
        default=False, description="Use eager execution for better memory management"
    )

    # Performance Optimization
    max_num_seqs: int = Field(
        default=16, description="Maximum number of sequences in a batch"
    )
    max_paddings: int = Field(default=512, description="Maximum padding tokens")
    tensor_parallel_size: int = Field(default=1, description="Tensor parallelism size")
    pipeline_parallel_size: int = Field(
        default=1, description="Pipeline parallelism size"
    )

    # FlashInfer Configuration
    attention_backend: str = Field(
        default="FLASHINFER", description="Use FlashInfer for optimized attention"
    )
    enable_chunked_prefill: bool = Field(
        default=True, description="Enable chunked prefill for memory efficiency"
    )
    max_num_batched_tokens: int = Field(
        default=8192, description="Maximum batched tokens for prefill"
    )

    # Sampling Configuration
    default_temperature: float = Field(
        default=0.1, description="Default sampling temperature"
    )
    default_top_p: float = Field(default=0.9, description="Default top-p sampling")
    default_max_tokens: int = Field(
        default=2048, description="Default maximum tokens to generate"
    )


class VLLMBackend:
    """vLLM backend implementation with FP8 quantization support."""

    def __init__(self, config: VLLMConfig | None = None):
        """Initialize vLLM backend with configuration.

        Args:
            config: vLLM configuration. If None, uses default settings.
        """
        self.config = config or VLLMConfig()
        self._llm: LLM | None = None
        self._async_engine: AsyncLLMEngine | None = None
        self._is_initialized = False

        logger.info(f"Initializing vLLM backend with model: {self.config.model_name}")
        logger.info(f"FP8 quantization: {self.config.quantization}")
        logger.info(f"KV cache dtype: {self.config.kv_cache_dtype}")
        logger.info(f"Max model length: {self.config.max_model_len}")

    def initialize(self) -> None:
        """Initialize the vLLM engine with FP8 optimization."""
        if self._is_initialized:
            logger.info("vLLM backend already initialized")
            return

        try:
            # Configure vLLM engine arguments for FP8 optimization
            engine_args = {
                "model": self.config.model_path or self.config.model_name,
                "quantization": self.config.quantization,
                "kv_cache_dtype": self.config.kv_cache_dtype,
                "max_model_len": self.config.max_model_len,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "enforce_eager": self.config.enforce_eager,
                "max_num_seqs": self.config.max_num_seqs,
                "max_paddings": self.config.max_paddings,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "pipeline_parallel_size": self.config.pipeline_parallel_size,
                "attention_backend": self.config.attention_backend,
                "enable_chunked_prefill": self.config.enable_chunked_prefill,
                "max_num_batched_tokens": self.config.max_num_batched_tokens,
                "trust_remote_code": True,  # Required for Qwen models
            }

            logger.info("Creating vLLM engine with FP8 optimization...")
            logger.info(f"Engine args: {engine_args}")

            # Initialize synchronous LLM engine
            self._llm = LLM(**engine_args)

            self._is_initialized = True
            logger.info("vLLM backend initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM backend: {e}")
            raise RuntimeError(f"vLLM initialization failed: {e}") from e

    async def initialize_async(self) -> None:
        """Initialize async vLLM engine for streaming operations."""
        if self._async_engine is not None:
            logger.info("Async vLLM engine already initialized")
            return

        try:
            from vllm.engine.arg_utils import AsyncEngineArgs

            # Configure async engine arguments
            engine_args = AsyncEngineArgs(
                model=self.config.model_path or self.config.model_name,
                quantization=self.config.quantization,
                kv_cache_dtype=self.config.kv_cache_dtype,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                enforce_eager=self.config.enforce_eager,
                max_num_seqs=self.config.max_num_seqs,
                max_paddings=self.config.max_paddings,
                tensor_parallel_size=self.config.tensor_parallel_size,
                pipeline_parallel_size=self.config.pipeline_parallel_size,
                attention_backend=self.config.attention_backend,
                enable_chunked_prefill=self.config.enable_chunked_prefill,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                trust_remote_code=True,
            )

            logger.info("Creating async vLLM engine...")
            self._async_engine = AsyncLLMEngine.from_engine_args(engine_args)

            logger.info("Async vLLM engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize async vLLM engine: {e}")
            raise RuntimeError(f"Async vLLM initialization failed: {e}") from e

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate text using vLLM backend.

        Args:
            prompts: Input prompts (single string or list)
            sampling_params: Sampling parameters for generation
            **kwargs: Additional generation parameters

        Returns:
            List of generated text responses

        Raises:
            RuntimeError: If backend is not initialized
        """
        if not self._is_initialized or self._llm is None:
            raise RuntimeError("vLLM backend not initialized. Call initialize() first.")

        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Configure sampling parameters
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", self.config.default_temperature),
                top_p=kwargs.get("top_p", self.config.default_top_p),
                max_tokens=kwargs.get("max_tokens", self.config.default_max_tokens),
                stop=kwargs.get("stop_sequences"),
            )

        try:
            logger.debug(f"Generating responses for {len(prompts)} prompts")
            outputs = self._llm.generate(prompts, sampling_params)

            # Extract generated text from outputs
            responses = []
            for output in outputs:
                if output.outputs:
                    responses.append(output.outputs[0].text)
                else:
                    responses.append("")

            logger.debug(f"Generated {len(responses)} responses")
            return responses

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"vLLM generation failed: {e}") from e

    async def generate_stream(
        self, prompt: str, sampling_params: SamplingParams | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using async vLLM backend.

        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters for generation
            **kwargs: Additional generation parameters

        Yields:
            Generated text chunks

        Raises:
            RuntimeError: If async backend is not initialized
        """
        if self._async_engine is None:
            await self.initialize_async()

        # Configure sampling parameters
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", self.config.default_temperature),
                top_p=kwargs.get("top_p", self.config.default_top_p),
                max_tokens=kwargs.get("max_tokens", self.config.default_max_tokens),
                stop=kwargs.get("stop_sequences"),
            )

        try:
            logger.debug(f"Starting streaming generation for prompt: {prompt[:50]}...")

            async for request_output in self._async_engine.generate(
                prompt, sampling_params, request_id=None
            ):
                if request_output.outputs:
                    # Extract the latest generated text chunk
                    output = request_output.outputs[0]
                    yield output.text

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise RuntimeError(f"vLLM streaming generation failed: {e}") from e

    def get_model_info(self) -> dict[str, Any]:
        """Get model information and configuration.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.config.model_name,
            "model_path": self.config.model_path,
            "quantization": self.config.quantization,
            "kv_cache_dtype": self.config.kv_cache_dtype,
            "max_model_len": self.config.max_model_len,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "attention_backend": self.config.attention_backend,
            "is_initialized": self._is_initialized,
        }

    def cleanup(self) -> None:
        """Clean up vLLM resources."""
        try:
            if self._llm is not None:
                # Note: vLLM doesn't have explicit cleanup method
                # GPU memory will be freed when the object is deleted
                self._llm = None
                logger.info("Synchronous vLLM engine cleaned up")

            if self._async_engine is not None:
                # Async engine cleanup
                self._async_engine = None
                logger.info("Async vLLM engine cleaned up")

            self._is_initialized = False

        except Exception as e:
            logger.warning(f"Error during vLLM cleanup: {e}")


# Global vLLM backend instance
_global_vllm_backend: VLLMBackend | None = None


def get_vllm_backend() -> VLLMBackend:
    """Get or create global vLLM backend instance.

    Returns:
        Global vLLM backend instance
    """
    global _global_vllm_backend

    if _global_vllm_backend is None:
        # Create configuration from settings
        config = VLLMConfig(
            model_name=settings.model_name,
            quantization="fp8",  # Force FP8 quantization
            kv_cache_dtype="fp8",  # FP8 KV cache
            max_model_len=min(settings.context_window_size, 131072),  # 128K max
            gpu_memory_utilization=0.85,  # Conservative for 16GB VRAM
            default_temperature=settings.llm_temperature,
            default_max_tokens=settings.llm_max_tokens,
        )

        _global_vllm_backend = VLLMBackend(config)
        logger.info("Created global vLLM backend instance")

    return _global_vllm_backend


def initialize_vllm_backend() -> VLLMBackend:
    """Initialize and return the global vLLM backend.

    Returns:
        Initialized vLLM backend instance

    Raises:
        RuntimeError: If initialization fails
    """
    backend = get_vllm_backend()
    backend.initialize()
    return backend


async def initialize_vllm_backend_async() -> VLLMBackend:
    """Initialize and return the global async vLLM backend.

    Returns:
        Initialized vLLM backend instance with async support

    Raises:
        RuntimeError: If initialization fails
    """
    backend = get_vllm_backend()
    backend.initialize()  # Initialize sync first
    await backend.initialize_async()  # Then initialize async
    return backend


def cleanup_vllm_backend() -> None:
    """Clean up the global vLLM backend."""
    global _global_vllm_backend

    if _global_vllm_backend is not None:
        _global_vllm_backend.cleanup()
        _global_vllm_backend = None
        logger.info("Global vLLM backend cleaned up")
