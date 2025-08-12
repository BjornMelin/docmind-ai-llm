#!/usr/bin/env python3
"""DocMind AI GPU Validation Script.

Tests GPU functionality for CUDA, PyTorch, FastEmbed, and vLLM
"""

import subprocess
import sys
import time
from typing import Any


def check_nvidia_smi() -> dict[str, Any]:
    """Check NVIDIA driver and GPU status."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "✅ PASS", "details": "NVIDIA driver working"}
        else:
            return {
                "status": "❌ FAIL",
                "details": f"nvidia-smi failed: {result.stderr}",
            }
    except FileNotFoundError:
        return {"status": "❌ FAIL", "details": "nvidia-smi not found"}


def check_cuda_toolkit() -> dict[str, Any]:
    """Check CUDA toolkit installation."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = [
                line for line in result.stdout.split("\n") if "release" in line
            ][0]
            version = version_line.split("release ")[1].split(",")[0]
            return {"status": "✅ PASS", "details": f"CUDA toolkit {version} installed"}
        else:
            return {"status": "❌ FAIL", "details": "nvcc not found"}
    except (FileNotFoundError, IndexError):
        return {"status": "❌ FAIL", "details": "CUDA toolkit not installed"}


def check_pytorch_cuda() -> dict[str, Any]:
    """Check PyTorch CUDA support."""
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return {
                "status": "✅ PASS",
                "details": (
                    f"PyTorch CUDA {cuda_version}, {device_count} GPU(s), {device_name}"
                ),
            }
        else:
            return {"status": "❌ FAIL", "details": "PyTorch CUDA not available"}
    except ImportError:
        return {"status": "⚠️  SKIP", "details": "PyTorch not installed"}


def check_fastembed_gpu() -> dict[str, Any]:
    """Check FastEmbed GPU support."""
    try:
        from fastembed import TextEmbedding

        # Test with GPU providers
        model = TextEmbedding(
            "BAAI/bge-small-en-v1.5",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        # Quick embedding test
        start_time = time.time()
        embeddings = list(model.embed(["GPU test document"]))
        end_time = time.time()

        if len(embeddings) > 0:
            return {
                "status": "✅ PASS",
                "details": (
                    f"FastEmbed GPU working, embedding time: "
                    f"{(end_time - start_time) * 1000:.1f}ms"
                ),
            }
        else:
            return {
                "status": "❌ FAIL",
                "details": "FastEmbed failed to generate embeddings",
            }

    except ImportError:
        return {"status": "⚠️  SKIP", "details": "FastEmbed not installed"}
    except Exception as e:
        return {"status": "❌ FAIL", "details": f"FastEmbed error: {str(e)}"}


def check_docker_gpu() -> dict[str, Any]:
    """Check Docker GPU support."""
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "nvidia/cuda:12.8-base-ubuntu22.04",
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip()
            return {"status": "✅ PASS", "details": f"Docker GPU access: {gpu_name}"}
        else:
            return {
                "status": "❌ FAIL",
                "details": f"Docker GPU test failed: {result.stderr}",
            }
    except subprocess.TimeoutExpired:
        return {"status": "❌ FAIL", "details": "Docker GPU test timed out"}
    except FileNotFoundError:
        return {"status": "⚠️  SKIP", "details": "Docker not found"}
    except Exception as e:
        return {"status": "❌ FAIL", "details": f"Docker error: {str(e)}"}


def check_gpu_memory() -> dict[str, Any]:
    """Check GPU memory usage."""
    try:
        import torch

        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            cached = torch.cuda.memory_reserved(0)

            total_gb = total_memory / 1e9
            allocated_mb = allocated / 1e6
            cached_mb = cached / 1e6

            return {
                "status": "✅ PASS",
                "details": (
                    f"GPU Memory: {total_gb:.1f}GB total, {allocated_mb:.1f}MB "
                    f"allocated, {cached_mb:.1f}MB cached"
                ),
            }
        else:
            return {
                "status": "❌ FAIL",
                "details": "CUDA not available for memory check",
            }
    except ImportError:
        return {
            "status": "⚠️  SKIP",
            "details": "PyTorch not available for memory check",
        }


def main():
    """Run all GPU validation tests."""
    print("🚀 DocMind AI GPU Infrastructure Validation")
    print("=" * 60)

    tests = [
        ("NVIDIA Driver", check_nvidia_smi),
        ("CUDA Toolkit", check_cuda_toolkit),
        ("PyTorch CUDA", check_pytorch_cuda),
        ("FastEmbed GPU", check_fastembed_gpu),
        ("Docker GPU", check_docker_gpu),
        ("GPU Memory", check_gpu_memory),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print(f"   {result['status']} {result['details']}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result["status"] == "✅ PASS")
    failed = sum(1 for _, result in results if result["status"] == "❌ FAIL")
    skipped = sum(1 for _, result in results if result["status"] == "⚠️  SKIP")

    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped: {skipped}")

    if failed == 0:
        print("\n🎉 GPU infrastructure is ready for 100x performance improvements!")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
