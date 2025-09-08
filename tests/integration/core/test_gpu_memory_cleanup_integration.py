"""GPU memory cleanup integration test (CUDA required).

This test exercises real CUDA memory allocation/cleanup behavior and is marked
as an integration test to avoid running in unit-tier environments.
"""

import pytest


@pytest.mark.integration
@pytest.mark.requires_gpu
@pytest.mark.skipif(
    __import__("torch").cuda.is_available() is False, reason="CUDA not available"
)
def test_gpu_memory_cleanup_integration():
    import torch

    torch.cuda.synchronize()
    initial_memory = torch.cuda.memory_allocated()

    # Allocate on GPU and verify memory increased
    large_tensor = torch.randn(1000, 1000, device="cuda")
    after_allocation = torch.cuda.memory_allocated()
    assert after_allocation > initial_memory

    # Cleanup
    del large_tensor
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    final_memory = torch.cuda.memory_allocated()
    assert final_memory <= after_allocation
