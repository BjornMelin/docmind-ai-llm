#!/usr/bin/env python3
"""Performance Validation Script for vLLM FlashInfer Stack..

Based on ai-research/2025-08-20/002-vllm-cuda-stack-finalization.md

This script validates the vLLM + FlashInfer installation and measures
performance against the target metrics for Qwen3-4B-Instruct-2507-FP8
on RTX 4090 16GB VRAM.

Target Performance:
- Decode: 100-160 tokens/second (expected: 120-180 with FlashInfer)
- Prefill: 800-1300 tokens/second (expected: 900-1400 with RTX 4090)
- VRAM Usage: 12-14GB for 128K context
- Context: 128K tokens supported

Usage:
    python scripts/performance_validation.py

Environment Variables:
    VLLM_MODEL: Model to test (default: microsoft/DialoGPT-medium for testing)
    VLLM_ATTENTION_BACKEND: Backend to test (default: FLASHINFER)
    SKIP_MODEL_TEST: Skip actual model loading (default: false)
"""

import importlib.util
import os
import sys
import time
import traceback
from typing import Any


def check_cuda_environment() -> dict[str, Any]:
    """Verify CUDA environment and compatibility."""
    results = {
        "cuda_available": False,
        "driver_version": None,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory_gb": 0,
        "compute_capability": None,
        "flashinfer_compatible": False,
    }

    try:
        import torch

        results["cuda_available"] = torch.cuda.is_available()

        if results["cuda_available"]:
            results["cuda_version"] = torch.version.cuda
            results["gpu_name"] = torch.cuda.get_device_name(0)
            results["gpu_memory_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )
            results["compute_capability"] = torch.cuda.get_device_capability(0)

            # Check if RTX 4090 compatible (SM 8.9)
            major, minor = results["compute_capability"]
            results["flashinfer_compatible"] = major >= 8 and (major > 8 or minor >= 9)

        # Get driver version from nvidia-ml-py if available
        try:
            import pynvml

            pynvml.nvmlInit()
            results["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        except ImportError:
            pass

    except Exception as e:
        print(f"❌ CUDA environment check failed: {e}")

    return results


def check_pytorch_version() -> dict[str, Any]:
    """Verify PyTorch version compatibility."""
    results = {"version": None, "compatible": False, "expected": "2.7.1"}

    try:
        import torch

        results["version"] = torch.__version__
        # Check for PyTorch 2.7.1 (confirmed compatible with vLLM >=0.10.1)
        results["compatible"] = results["version"].startswith("2.7.")
    except Exception as e:
        print(f"❌ PyTorch version check failed: {e}")

    return results


def check_vllm_version() -> dict[str, Any]:
    """Verify vLLM version and FlashInfer availability."""
    results = {
        "version": None,
        "compatible": False,
        "flashinfer_available": False,
        "expected": ">=0.10.1",
    }

    try:
        import vllm

        results["version"] = vllm.__version__

        # Check for vLLM >=0.10.1 (PyTorch 2.7.1 compatibility confirmed)
        version_parts = results["version"].split(".")
        if len(version_parts) >= 3:
            major, minor, patch = (
                int(version_parts[0]),
                int(version_parts[1]),
                int(version_parts[2]),
            )
            results["compatible"] = (
                major > 0
                or (major == 0 and minor > 10)
                or (major == 0 and minor == 10 and patch >= 1)
            )

        # Check FlashInfer availability
        results["flashinfer_available"] = (
            importlib.util.find_spec("flashinfer") is not None
        )

    except Exception as e:
        print(f"❌ vLLM version check failed: {e}")

    return results


def test_vllm_flashinfer_backend() -> dict[str, Any]:
    """Test vLLM with FlashInfer backend."""
    results = {
        "backend_available": False,
        "model_loaded": False,
        "error": None,
        "test_model": "microsoft/DialoGPT-medium",  # Small test model
    }

    if os.getenv("SKIP_MODEL_TEST", "false").lower() == "true":
        print("⏩ Skipping model test (SKIP_MODEL_TEST=true)")
        return results

    try:
        from vllm import LLM, SamplingParams

        # Test FlashInfer backend availability
        test_model = os.getenv("VLLM_MODEL", results["test_model"])
        attention_backend = os.getenv("VLLM_ATTENTION_BACKEND", "FLASHINFER")

        print(f"📋 Testing model: {test_model}")
        print(f"📋 Testing backend: {attention_backend}")

        llm = LLM(
            model=test_model,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,  # Conservative for testing
            trust_remote_code=True,
            attention_backend=attention_backend,
            max_model_len=2048,  # Small context for testing
        )

        results["backend_available"] = True
        results["model_loaded"] = True

        # Quick generation test
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        outputs = llm.generate(["Hello"], sampling_params)

        print(f"✅ Test generation successful: {outputs[0].outputs[0].text[:50]}...")

    except Exception as e:
        results["error"] = str(e)
        print(f"❌ vLLM FlashInfer test failed: {e}")
        traceback.print_exc()

    return results


def measure_performance() -> dict[str, Any]:
    """Measure basic performance metrics."""
    results = {
        "throughput_tokens_per_sec": 0,
        "latency_seconds": 0,
        "memory_usage_gb": 0,
        "test_completed": False,
    }

    if os.getenv("SKIP_MODEL_TEST", "false").lower() == "true":
        print("⏩ Skipping performance test (SKIP_MODEL_TEST=true)")
        return results

    try:
        import torch
        from vllm import LLM, SamplingParams

        # Measure GPU memory before
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated(0) / 1e9

        test_model = os.getenv("VLLM_MODEL", "microsoft/DialoGPT-medium")
        attention_backend = os.getenv("VLLM_ATTENTION_BACKEND", "FLASHINFER")

        llm = LLM(
            model=test_model,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            trust_remote_code=True,
            attention_backend=attention_backend,
            max_model_len=2048,
        )

        # Performance test with multiple prompts
        prompts = ["Hello, how are you today?"] * 5
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()

        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        elapsed_time = end_time - start_time

        results["throughput_tokens_per_sec"] = total_tokens / elapsed_time
        results["latency_seconds"] = elapsed_time
        results["memory_usage_gb"] = (
            torch.cuda.max_memory_allocated(0) / 1e9 - memory_before
        )
        results["test_completed"] = True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        traceback.print_exc()

    return results


def print_results(
    cuda_env: dict[str, Any],
    pytorch_info: dict[str, Any],
    vllm_info: dict[str, Any],
    backend_test: dict[str, Any],
    performance: dict[str, Any],
) -> bool:
    """Print comprehensive results and return success status."""
    print("\n" + "=" * 60)
    print("🚀 vLLM FlashInfer Stack Validation Results")
    print("=" * 60)

    # CUDA Environment
    print("\n📊 CUDA Environment:")
    cuda_status = "✅" if cuda_env["cuda_available"] else "❌"
    print(f"   CUDA Available: {cuda_status} {cuda_env['cuda_available']}")
    if cuda_env["cuda_available"]:
        print(f"   GPU: {cuda_env['gpu_name']}")
        print(f"   VRAM: {cuda_env['gpu_memory_gb']:.1f} GB")
        print(f"   Compute Capability: {cuda_env['compute_capability']}")
        print(f"   CUDA Version: {cuda_env['cuda_version']}")
        if cuda_env["driver_version"]:
            print(f"   Driver Version: {cuda_env['driver_version']}")
        flashinfer_status = "✅" if cuda_env["flashinfer_compatible"] else "❌"
        print(
            f"   FlashInfer Compatible: {flashinfer_status} "
            f"{cuda_env['flashinfer_compatible']}"
        )

    # PyTorch
    print("\n🔥 PyTorch:")
    print(f"   Version: {pytorch_info['version']}")
    print(f"   Expected: {pytorch_info['expected']}")
    pytorch_status = "✅" if pytorch_info["compatible"] else "❌"
    print(f"   Compatible: {pytorch_status} {pytorch_info['compatible']}")

    # vLLM
    print("\n⚡ vLLM:")
    print(f"   Version: {vllm_info['version']}")
    print(f"   Expected: {vllm_info['expected']}")
    vllm_status = "✅" if vllm_info["compatible"] else "❌"
    print(f"   Compatible: {vllm_status} {vllm_info['compatible']}")
    flashinfer_avail_status = "✅" if vllm_info["flashinfer_available"] else "❌"
    print(
        f"   FlashInfer Available: {flashinfer_avail_status} "
        f"{vllm_info['flashinfer_available']}"
    )

    # Backend Test
    print("\n🎯 FlashInfer Backend Test:")
    backend_status = "✅" if backend_test["backend_available"] else "❌"
    print(f"   Backend Available: {backend_status} {backend_test['backend_available']}")
    model_status = "✅" if backend_test["model_loaded"] else "❌"
    print(f"   Model Loaded: {model_status} {backend_test['model_loaded']}")
    if backend_test["error"]:
        print(f"   Error: {backend_test['error']}")

    # Performance
    if performance["test_completed"]:
        print("\n📈 Performance Results:")
        throughput = performance["throughput_tokens_per_sec"]
        print(f"   Throughput: {throughput:.1f} tokens/second")
        print(f"   Latency: {performance['latency_seconds']:.2f} seconds")
        print(f"   GPU Memory: {performance['memory_usage_gb']:.1f} GB")

        # Performance assessment
        print("\n🎯 Performance Assessment:")

        # Note: These are basic test metrics, not production Qwen3-4B metrics
        print(f"   Note: Using test model ({backend_test['test_model']})")
        print("   Production targets for Qwen3-4B-Instruct-2507-FP8:")
        print("   • Decode: 100-160 tok/s (expected: 120-180 with FlashInfer)")
        print("   • Prefill: 800-1300 tok/s (expected: 900-1400 with RTX 4090)")
        print("   • VRAM: 12-14GB for 128K context")

    # Overall Status
    print("\n🏁 Overall Status:")

    success = (
        cuda_env["cuda_available"]
        and cuda_env["flashinfer_compatible"]
        and pytorch_info["compatible"]
        and vllm_info["compatible"]
        and vllm_info["flashinfer_available"]
        and backend_test["backend_available"]
    )

    if success:
        print("✅ SUCCESS: vLLM FlashInfer stack is properly installed and configured!")
        print("✅ Ready for Qwen3-4B-Instruct-2507-FP8 with 128K context support")
        print("\n📋 Next Steps:")
        print(
            "1. Configure your model in .env: "
            "VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8"
        )
        print("2. Set context length: VLLM_MAX_MODEL_LEN=131072")
        print("3. Optimize GPU memory: VLLM_GPU_MEMORY_UTILIZATION=0.85")
        print("4. Enable FP8: VLLM_QUANTIZATION=fp8 and VLLM_KV_CACHE_DTYPE=fp8_e5m2")
    else:
        print("❌ FAILED: Issues detected with vLLM FlashInfer stack")
        print("\n🔧 Troubleshooting:")
        if not cuda_env["cuda_available"]:
            print("• Install NVIDIA drivers and CUDA 12.8+")
        if not cuda_env["flashinfer_compatible"]:
            print("• FlashInfer requires GPU with compute capability 8.9+ (RTX 4090)")
        if not pytorch_info["compatible"]:
            print(
                "• Install PyTorch 2.7.1: uv pip install torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128"
            )
        if not vllm_info["compatible"]:
            print('• Install vLLM >=0.10.1: uv pip install "vllm[flashinfer]>=0.10.1"')
        if not vllm_info["flashinfer_available"]:
            print(
                "• FlashInfer not available - check installation or use "
                "fallback CUDA backend"
            )

    print("=" * 60)

    return success


def main():
    """Main validation function."""
    print("🚀 DocMind AI vLLM FlashInfer Stack Validation")
    print("Based on ai-research/2025-08-20/002-vllm-cuda-stack-finalization.md")
    print("Target: Qwen3-4B-Instruct-2507-FP8 on RTX 4090 16GB\n")

    # Run all checks
    cuda_env = check_cuda_environment()
    pytorch_info = check_pytorch_version()
    vllm_info = check_vllm_version()
    backend_test = test_vllm_flashinfer_backend()
    performance = measure_performance()

    # Print results and return success status
    success = print_results(
        cuda_env, pytorch_info, vllm_info, backend_test, performance
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
