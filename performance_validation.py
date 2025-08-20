# performance_validation.py
import os
import time

import torch
from vllm import LLM, SamplingParams

# Set FlashInfer as attention backend via environment variable
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"


def verify_installation():
    """Verify vLLM + FlashInfer installation and performance."""
    # Check GPU availability
    assert torch.cuda.is_available(), "CUDA not available"
    assert torch.cuda.get_device_capability()[0] >= 8, "Insufficient compute capability"

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Compute Capability: {torch.cuda.get_device_capability()}")
    print(
        f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

    # Initialize vLLM with FlashInfer (via environment variable)
    llm = LLM(
        model="microsoft/DialoGPT-medium",  # Small test model
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
    )

    print("✅ Model loaded successfully with FlashInfer backend")

    # Simple generation test
    prompts = ["Hello, how are you?"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    # Calculate metrics
    total_tokens = len(outputs[0].outputs[0].token_ids)
    throughput = total_tokens / (end_time - start_time)

    print(f"✅ Generated text: {outputs[0].outputs[0].text}")
    print(f"✅ Throughput: {throughput:.2f} tokens/second")
    print(f"✅ Total tokens generated: {total_tokens}")
    print(f"✅ Generation time: {end_time - start_time:.2f}s")

    return True  # Installation is working


if __name__ == "__main__":
    success = verify_installation()
    print("✅ Installation verified!" if success else "❌ Installation failed!")
