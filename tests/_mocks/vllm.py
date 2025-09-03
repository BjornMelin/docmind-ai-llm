"""Mock vLLM classes for testing."""

from typing import Any

from src.config import settings

# Mock availability flags for test compatibility
VLLM_AVAILABLE = True


class MockVLLMConfig:
    """Mock VLLMConfig for testing."""

    def __init__(self, **kwargs):
        """Initialize mock VLLMConfig."""
        self.model = kwargs.get("model", settings.model_name)
        self.max_model_len = kwargs.get("max_model_len", settings.context_window_size)
        self.kv_cache_dtype = kwargs.get("kv_cache_dtype", settings.kv_cache_dtype)
        self.attention_backend = kwargs.get(
            "attention_backend", settings.vllm_attention_backend
        )
        self.gpu_memory_utilization = kwargs.get(
            "gpu_memory_utilization", settings.vllm_gpu_memory_utilization
        )
        self.enable_chunked_prefill = kwargs.get(
            "enable_chunked_prefill", settings.vllm_enable_chunked_prefill
        )
        self.max_num_seqs = kwargs.get("max_num_seqs", settings.vllm_max_num_seqs)
        self.max_num_batched_tokens = kwargs.get(
            "max_num_batched_tokens", settings.vllm_max_num_batched_tokens
        )


class MockContextManager:
    """Mock ContextManager for testing."""

    def __init__(self, max_context_tokens: int = 131072):
        """Initialize mock context manager."""
        self.max_context_tokens = max_context_tokens
        self.trim_threshold = int(max_context_tokens * 0.9)

    def estimate_tokens(self, messages: list) -> int:
        """Estimate token count."""
        if not messages:
            return 0
        total_chars = sum(len(str(msg)) for msg in messages)
        return total_chars // 4


class MockVLLMManager:
    """Mock VLLMManager for testing."""

    def __init__(self, config):
        """Initialize mock VLLMManager."""
        self.config = config
        self.llm = None
        self.context_manager = MockContextManager()

    def initialize_engine(self) -> bool:
        """Mock engine initialization."""
        return True

    def validate_performance(self) -> dict[str, Any]:
        """Mock performance validation."""
        return {"validation_passed": True}


# Factory functions for testing compatibility
def create_mock_vllm_config(**kwargs) -> MockVLLMConfig:
    """Create mock VLLMConfig for tests."""
    return MockVLLMConfig(**kwargs)


def create_mock_vllm_manager(config=None) -> MockVLLMManager:
    """Create mock VLLMManager for tests."""
    if config is None:
        config = create_mock_vllm_config()
    return MockVLLMManager(config)


# Mock vLLM classes that tests might need
class MockAsyncLLMEngine:
    """Mock AsyncLLMEngine for testing."""

    @classmethod
    def from_engine_args(cls, *args, **kwargs):
        """Mock factory method."""
        return cls()


class MockLLM:
    """Mock LLM class for testing."""

    def __init__(self, *args, **kwargs):
        """Initialize mock LLM."""
        pass


class MockSamplingParams:
    """Mock SamplingParams for testing."""

    def __init__(self, *args, **kwargs):
        """Initialize mock sampling params."""
        pass


# Exports for testing
__all__ = [
    "VLLM_AVAILABLE",
    "MockVLLMConfig",
    "MockContextManager",
    "MockVLLMManager",
    "MockAsyncLLMEngine",
    "MockLLM",
    "MockSamplingParams",
    "create_mock_vllm_config",
    "create_mock_vllm_manager",
]
