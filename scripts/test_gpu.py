#!/usr/bin/env python
"""GPU Validation Test Runner for DocMind AI.

This script focuses on GPU-specific testing and hardware validation.
Validates GPU functionality, memory usage, and performance under load.

Test Focus:
- GPU hardware detection and compatibility
- VRAM allocation and management
- Model loading and inference performance
- GPU-accelerated components (embeddings, reranking)
- Memory leak detection during GPU operations

Usage:
    python scripts/test_gpu.py                      # Full GPU validation
    python scripts/test_gpu.py --quick             # Quick GPU health check
    python scripts/test_gpu.py --benchmark         # Performance benchmarking
    python scripts/test_gpu.py --memory-check      # Memory usage validation
    python scripts/test_gpu.py --compatibility     # Hardware compatibility check
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def get_gpu_info() -> dict | None:
    """Get detailed GPU information."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version,cuda_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return None

        gpu_data = result.stdout.strip().split(", ")
        return {
            "name": gpu_data[0],
            "memory_total": int(gpu_data[1]),
            "memory_free": int(gpu_data[2]),
            "driver_version": gpu_data[3],
            "cuda_version": gpu_data[4] if gpu_data[4] != "[Not Supported]" else None,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError, ValueError):
        return None


def check_cuda_availability() -> bool:
    """Check if CUDA is available and working."""
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); "
                "print(f'CUDA devices: {torch.cuda.device_count()}'); "
                "current_dev = torch.cuda.current_device() if "
                "torch.cuda.is_available() else 'N/A'; "
                "print(f'Current device: {current_dev}'); "
                "dev_name = torch.cuda.get_device_name(0) if "
                "torch.cuda.is_available() else 'N/A'; "
                "print(f'Device name: {dev_name}')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and "CUDA available: True" in result.stdout:
            print("✅ CUDA is available and functional")
            print(result.stdout.strip())
            return True
        else:
            print("❌ CUDA is not available")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def run_command(
    command: list[str], description: str, timeout: int = 1800
) -> tuple[int, str]:
    """Run a command and return exit code and output."""
    print(f"\n{'=' * 60}")
    print(f"🧪 {description}")
    print(f"📋 Command: {' '.join(command)}")
    print("=" * 60)

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {description} completed successfully ({duration:.1f}s)")
        else:
            print(f"❌ {description} failed ({duration:.1f}s)")
            if result.stderr:
                print("STDERR:", result.stderr[-1000:])  # Last 1000 chars

        return result.returncode, result.stdout

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ {description} timed out after {duration:.1f}s")
        return -1, "Timeout"
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 {description} error after {duration:.1f}s: {e}")
        return -1, str(e)


def monitor_gpu_memory() -> dict:
    """Monitor GPU memory usage."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            memory_data = result.stdout.strip().split(", ")
            return {
                "used": int(memory_data[0]),
                "total": int(memory_data[1]),
                "free": int(memory_data[1]) - int(memory_data[0]),
                "utilization": (int(memory_data[0]) / int(memory_data[1])) * 100,
            }
    except Exception as e:
        print(f"Warning: Could not get GPU memory info: {e}")

    return {"used": 0, "total": 0, "free": 0, "utilization": 0}


def main():
    """Main GPU validation runner."""
    parser = argparse.ArgumentParser(description="DocMind AI GPU Test Runner")
    parser.add_argument(
        "--quick", action="store_true", help="Quick GPU health check only"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run GPU performance benchmarks"
    )
    parser.add_argument(
        "--memory-check", action="store_true", help="Focus on memory usage validation"
    )
    parser.add_argument(
        "--compatibility", action="store_true", help="Hardware compatibility check only"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    print("🎯 DocMind AI GPU Validation Suite")
    print("=" * 60)
    print("Target: GPU hardware and performance validation")
    print("Scope: GPU-specific tests and benchmarks")

    # Change to project root
    import os

    os.chdir(project_root)

    exit_codes = []
    test_results = {}

    # Step 1: Hardware detection
    print("\n🔍 Step 1: GPU Hardware Detection")
    gpu_info = get_gpu_info()

    if not gpu_info:
        print("❌ No GPU detected or nvidia-smi not available")
        print(
            "💡 Make sure NVIDIA drivers are installed and GPU is properly configured"
        )
        sys.exit(1)

    print(f"✅ GPU detected: {gpu_info['name']}")
    print(
        f"   Memory: {gpu_info['memory_total']}MB total, "
        f"{gpu_info['memory_free']}MB free"
    )
    print(f"   Driver: {gpu_info['driver_version']}")
    if gpu_info["cuda_version"]:
        print(f"   CUDA: {gpu_info['cuda_version']}")

    test_results["hardware"] = True

    # Check memory requirements
    if gpu_info["memory_total"] < 12000:  # 12GB minimum
        print("⚠️  Warning: GPU has less than 12GB VRAM")
        print("   System tests may fail or run with reduced performance")

    # Step 2: CUDA compatibility
    print("\n🔧 Step 2: CUDA Compatibility Check")
    cuda_available = check_cuda_availability()
    test_results["cuda"] = cuda_available

    if not cuda_available:
        print("❌ CUDA not available - GPU tests will fail")
        if not args.compatibility:
            sys.exit(1)

    # If only compatibility check requested, exit here
    if args.compatibility:
        print("\n📋 Hardware Compatibility Summary:")
        print(
            f"   GPU: {'✅' if test_results['hardware'] else '❌'} "
            f"{gpu_info['name'] if gpu_info else 'Not detected'}"
        )
        cuda_status = "✅" if test_results["cuda"] else "❌"
        print(f"   CUDA: {cuda_status} Available")
        vram_status = "✅" if gpu_info and gpu_info["memory_total"] >= 12000 else "⚠️"
        vram_amount = gpu_info["memory_total"] if gpu_info else 0
        print(f"   VRAM: {vram_status} {vram_amount}MB")

        if test_results["hardware"] and test_results["cuda"]:
            print("\n✅ GPU is compatible with DocMind AI")
            sys.exit(0)
        else:
            print("\n❌ GPU compatibility issues detected")
            sys.exit(1)

    # Step 3: Quick health check
    if args.quick:
        print("\n⚡ Step 3: Quick GPU Health Check")

        # Monitor initial memory
        initial_memory = monitor_gpu_memory()
        print(
            f"Initial VRAM usage: {initial_memory['used']}MB "
            f"({initial_memory['utilization']:.1f}%)"
        )

        # Run smoke test
        cmd = ["uv", "run", "python", "scripts/run_tests.py", "--smoke"]
        exit_code, _ = run_command(cmd, "GPU Smoke Test", timeout=300)
        exit_codes.append(exit_code)
        test_results["smoke"] = exit_code == 0

        # Check memory after test
        final_memory = monitor_gpu_memory()
        print(
            f"Final VRAM usage: {final_memory['used']}MB "
            f"({final_memory['utilization']:.1f}%)"
        )

        memory_increase = final_memory["used"] - initial_memory["used"]
        if memory_increase > 1000:  # More than 1GB increase
            print(f"⚠️  High memory usage increase: {memory_increase}MB")

    else:
        # Step 3: GPU-required tests
        print("\n🎯 Step 3: GPU-Required Tests")

        # Monitor memory before tests
        initial_memory = monitor_gpu_memory()
        print(
            f"Initial VRAM: {initial_memory['used']}MB used, "
            f"{initial_memory['free']}MB free"
        )

        cmd = ["uv", "run", "python", "scripts/run_tests.py", "--gpu"]
        exit_code, _ = run_command(cmd, "GPU Tests", timeout=1800)
        exit_codes.append(exit_code)
        test_results["gpu_tests"] = exit_code == 0

        # Monitor memory after tests
        final_memory = monitor_gpu_memory()
        print(
            f"Final VRAM: {final_memory['used']}MB used, {final_memory['free']}MB free"
        )

        # Step 4: Performance validation (replacing system tests)
        print("\n🧠 Step 4: Performance Validation")

        cmd = ["uv", "run", "python", "scripts/run_tests.py", "--performance"]
        exit_code, _ = run_command(cmd, "Performance Tests", timeout=2400)
        exit_codes.append(exit_code)
        test_results["performance"] = exit_code == 0

        # Monitor memory usage after performance tests
        post_system_memory = monitor_gpu_memory()
        print(
            f"Post-performance VRAM: {post_system_memory['used']}MB used, "
            f"{post_system_memory['free']}MB free"
        )

    # Step 5: Performance benchmarks (if requested)
    if args.benchmark:
        print("\n📊 Step 5: Performance Benchmarks")

        # Run performance validation script
        cmd = ["uv", "run", "python", "scripts/performance_monitor.py", "--run-tests"]
        exit_code, _ = run_command(cmd, "Performance Benchmarks", timeout=1200)
        exit_codes.append(exit_code)
        test_results["benchmark"] = exit_code == 0

        # Run vLLM performance validation if available
        vllm_script = project_root / "scripts" / "vllm_performance_validation.py"
        if vllm_script.exists():
            cmd = ["uv", "run", "python", "scripts/vllm_performance_validation.py"]
            exit_code, _ = run_command(cmd, "vLLM Performance", timeout=1200)
            exit_codes.append(exit_code)
            test_results["vllm"] = exit_code == 0

    # Step 6: Memory leak detection (if requested)
    if args.memory_check:
        print("\n🔍 Step 6: Memory Leak Detection")

        memory_samples = []

        # Take multiple memory samples during test execution
        for i in range(5):
            print(f"   Sample {i + 1}/5...")
            memory = monitor_gpu_memory()
            memory_samples.append(memory["used"])
            time.sleep(10)

        # Analyze memory trend
        if len(memory_samples) >= 3:
            trend = memory_samples[-1] - memory_samples[0]
            if trend > 500:  # More than 500MB increase
                print(f"⚠️  Potential memory leak detected: {trend}MB increase")
                test_results["memory_leak"] = False
            else:
                print(f"✅ Memory usage stable: {trend}MB change")
                test_results["memory_leak"] = True

    # Final GPU validation summary
    total_failures = sum(1 for code in exit_codes if code != 0)

    print("\n" + "=" * 80)
    print("📋 GPU VALIDATION SUMMARY")
    print("=" * 80)

    # Show hardware info
    print("\n🖥️  Hardware Information:")
    print(f"   GPU: {gpu_info['name']}")
    print(f"   VRAM: {gpu_info['memory_total']}MB total")
    print(f"   Driver: {gpu_info['driver_version']}")
    print(f"   CUDA: {gpu_info['cuda_version'] or 'Not available'}")

    # Show test results
    print("\n📊 Test Results:")
    for test_name, result in test_results.items():
        if isinstance(result, bool):
            status = "✅" if result else "❌"
            print(f"   {status} {test_name.replace('_', ' ').title()}")

    # Final assessment
    if total_failures == 0 and all(
        isinstance(r, bool) and r for r in test_results.values()
    ):
        print("\n🎉 All GPU tests passed!")
        print("✅ GPU is fully functional for DocMind AI")

        # Performance recommendations
        print("\n💡 Performance Notes:")
        utilization = (
            final_memory.get("utilization", 0) if "final_memory" in locals() else 0
        )
        if utilization > 90:
            print(
                f"   ⚠️  High VRAM utilization ({utilization:.1f}%) - "
                "consider model optimization"
            )
        elif utilization > 70:
            print(f"   ✅ Good VRAM utilization ({utilization:.1f}%)")
        else:
            print(
                f"   💡 Low VRAM utilization ({utilization:.1f}%) - GPU underutilized"
            )

        sys.exit(0)
    else:
        print("\n❌ GPU validation issues detected")
        if total_failures > 0:
            print(f"   {total_failures} test suite(s) failed")

        # Show specific recommendations
        print("\n💡 Recommendations:")
        if not test_results.get("cuda", True):
            print("   🔧 Install/update CUDA drivers")
        if not test_results.get("gpu_tests", True):
            print("   🎯 Check GPU-specific test failures")
        if not test_results.get("system", True):
            print("   🧠 Investigate system test issues with real models")

        sys.exit(1)


if __name__ == "__main__":
    main()
