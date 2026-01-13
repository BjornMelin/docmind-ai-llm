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
import csv
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from typing import NoReturn, TypedDict


class GPUInfo(TypedDict):
    """Structured GPU information returned by `nvidia-smi` queries."""

    name: str
    memory_total: int
    memory_free: int
    driver_version: str
    cuda_version: str | None


def get_gpu_info(*, cwd: Path | None = None) -> GPUInfo | None:
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
            check=False,
            timeout=10,
            cwd=cwd,
        )

        if result.returncode != 0:
            return None

        # `nvidia-smi` returns one CSV row per GPU; pick the first GPU for now.
        rows = list(csv.reader(StringIO(result.stdout.strip())))
        if not rows:
            return None

        row = [cell.strip() for cell in rows[0]]
        if len(row) < 5:
            return None

        cuda = row[4]
        gpu_info: GPUInfo = {
            "name": row[0],
            "memory_total": int(row[1]),
            "memory_free": int(row[2]),
            "driver_version": row[3],
            "cuda_version": None if cuda == "[Not Supported]" else cuda,
        }
        return gpu_info
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError, ValueError):
        return None


def check_cuda_availability(*, cwd: Path | None = None) -> bool:
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
            check=False,
            timeout=30,
            cwd=cwd,
        )

        if result.returncode == 0 and "CUDA available: True" in result.stdout:
            print("✅ CUDA is available and functional")
            print(result.stdout.strip())
            return True
        print("❌ CUDA is not available")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return False
    except (subprocess.TimeoutExpired, OSError, ValueError) as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def run_command(
    command: list[str],
    description: str,
    timeout: int = 1800,
    *,
    cwd: Path | None = None,
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
            check=False,
            timeout=timeout,
            cwd=cwd,
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
    except (OSError, ValueError) as e:
        duration = time.time() - start_time
        print(f"💥 {description} error after {duration:.1f}s: {e}")
        return -1, str(e)


def monitor_gpu_memory(*, cwd: Path | None = None) -> dict[str, float]:
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
            check=False,
            timeout=5,
            cwd=cwd,
        )

        if result.returncode == 0:
            rows = list(csv.reader(StringIO(result.stdout.strip())))
            if not rows:
                raise ValueError("Empty nvidia-smi output")
            row = [cell.strip() for cell in rows[0]]
            if len(row) < 2:
                raise ValueError("Unexpected nvidia-smi output")
            used = int(row[0])
            total = int(row[1])
            return {
                "used": float(used),
                "total": float(total),
                "free": float(total - used),
                "utilization": (used / total) * 100 if total else 0.0,
            }
    except (OSError, ValueError) as e:
        print(f"Warning: Could not get GPU memory info: {e}")

    return {"used": 0, "total": 0, "free": 0, "utilization": 0}


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for GPU validation."""
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
    return parser


def _print_header() -> None:
    """Print the GPU validation suite header."""
    print("🎯 DocMind AI GPU Validation Suite")
    print("=" * 60)
    print("Target: GPU hardware and performance validation")
    print("Scope: GPU-specific tests and benchmarks")


def _require_gpu_info(*, cwd: Path | None = None) -> GPUInfo:
    """Detect GPU hardware and return metadata or exit."""
    print("\n🔍 Step 1: GPU Hardware Detection")
    gpu_info = get_gpu_info(cwd=cwd)
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
    if gpu_info["memory_total"] < 12000:
        print("⚠️  Warning: GPU has less than 12GB VRAM")
        print("   System tests may fail or run with reduced performance")
    return gpu_info


def _check_cuda_compatibility_status(
    test_results: dict[str, bool],
    *,
    cwd: Path | None = None,
) -> bool:
    """Check CUDA availability and update results."""
    print("\n🔧 Step 2: CUDA Compatibility Check")
    cuda_available = check_cuda_availability(cwd=cwd)
    test_results["hardware"] = True
    test_results["cuda"] = cuda_available
    if not cuda_available:
        print("❌ CUDA not available - GPU tests will fail")
    return cuda_available


def _print_compatibility_summary_and_exit(
    gpu_info: GPUInfo,
    test_results: dict[str, bool],
) -> NoReturn:
    """Print hardware compatibility summary and exit."""
    print("\n📋 Hardware Compatibility Summary:")
    print(f"   GPU: {'✅' if test_results['hardware'] else '❌'} {gpu_info['name']}")
    cuda_status = "✅" if test_results["cuda"] else "❌"
    print(f"   CUDA: {cuda_status} Available")
    vram_status = "✅" if gpu_info["memory_total"] >= 12000 else "⚠️"
    print(f"   VRAM: {vram_status} {gpu_info['memory_total']}MB")
    if test_results["hardware"] and test_results["cuda"]:
        print("\n✅ GPU is compatible with DocMind AI")
        sys.exit(0)
    print("\n❌ GPU compatibility issues detected")
    sys.exit(1)


def _run_quick_check(
    exit_codes: list[int],
    test_results: dict[str, bool],
    *,
    cwd: Path | None = None,
) -> dict[str, float]:
    """Run the quick smoke test path and return final memory stats."""
    print("\n⚡ Step 3: Quick GPU Health Check")
    initial_memory = monitor_gpu_memory(cwd=cwd)
    print(
        f"Initial VRAM usage: {initial_memory['used']}MB "
        f"({initial_memory['utilization']:.1f}%)"
    )
    cmd = ["uv", "run", "python", "scripts/run_tests.py", "--smoke"]
    exit_code, _ = run_command(cmd, "GPU Smoke Test", timeout=300, cwd=cwd)
    exit_codes.append(exit_code)
    test_results["smoke"] = exit_code == 0
    final_memory = monitor_gpu_memory(cwd=cwd)
    print(
        f"Final VRAM usage: {final_memory['used']}MB "
        f"({final_memory['utilization']:.1f}%)"
    )
    memory_increase = final_memory["used"] - initial_memory["used"]
    if memory_increase > 1000:
        print(f"⚠️  High memory usage increase: {memory_increase}MB")
    return final_memory


def _run_gpu_tests(
    exit_codes: list[int],
    test_results: dict[str, bool],
    *,
    cwd: Path | None = None,
) -> dict[str, float]:
    """Run GPU-required tests and performance validation."""
    print("\n🎯 Step 3: GPU-Required Tests")
    initial_memory = monitor_gpu_memory(cwd=cwd)
    print(
        f"Initial VRAM: {initial_memory['used']}MB used, "
        f"{initial_memory['free']}MB free"
    )
    cmd = ["uv", "run", "python", "scripts/run_tests.py", "--gpu"]
    exit_code, _ = run_command(cmd, "GPU Tests", timeout=1800, cwd=cwd)
    exit_codes.append(exit_code)
    test_results["gpu_tests"] = exit_code == 0
    final_memory = monitor_gpu_memory(cwd=cwd)
    print(f"Final VRAM: {final_memory['used']}MB used, {final_memory['free']}MB free")
    print("\n🧠 Step 4: Performance Validation")
    cmd = ["uv", "run", "python", "scripts/run_tests.py", "--performance"]
    exit_code, _ = run_command(cmd, "Performance Tests", timeout=2400, cwd=cwd)
    exit_codes.append(exit_code)
    test_results["performance"] = exit_code == 0
    post_system_memory = monitor_gpu_memory(cwd=cwd)
    print(
        f"Post-performance VRAM: {post_system_memory['used']}MB used, "
        f"{post_system_memory['free']}MB free"
    )
    return post_system_memory


def _run_benchmarks(
    project_root: Path, exit_codes: list[int], test_results: dict[str, bool]
) -> None:
    """Run optional performance benchmark steps."""
    print("\n📊 Step 5: Performance Benchmarks")
    cmd = ["uv", "run", "python", "scripts/performance_monitor.py", "--run-tests"]
    exit_code, _ = run_command(
        cmd, "Performance Benchmarks", timeout=1200, cwd=project_root
    )
    exit_codes.append(exit_code)
    test_results["benchmark"] = exit_code == 0
    vllm_script = project_root / "scripts" / "vllm_performance_validation.py"
    if not vllm_script.exists():
        return
    cmd = ["uv", "run", "python", "scripts/vllm_performance_validation.py"]
    exit_code, _ = run_command(cmd, "vLLM Performance", timeout=1200, cwd=project_root)
    exit_codes.append(exit_code)
    test_results["vllm"] = exit_code == 0


def _run_memory_leak_check(
    test_results: dict[str, bool],
    *,
    cwd: Path | None = None,
) -> None:
    """Sample memory usage to detect potential leaks."""
    print("\n🔍 Step 6: Memory Leak Detection")
    memory_samples: list[float] = []
    for i in range(5):
        print(f"   Sample {i + 1}/5...")
        memory = monitor_gpu_memory(cwd=cwd)
        memory_samples.append(memory["used"])
        time.sleep(10)
    trend = memory_samples[-1] - memory_samples[0]
    if trend > 500:
        print(f"⚠️  Potential memory leak detected: {trend}MB increase")
        test_results["memory_leak"] = False
        return
    print(f"✅ Memory usage stable: {trend}MB change")
    test_results["memory_leak"] = True


def _print_summary(
    gpu_info: GPUInfo,
    test_results: dict[str, bool],
    total_failures: int,
    final_memory: dict[str, float] | None,
) -> NoReturn:
    """Print a final validation summary and exit with status."""
    print("\n" + "=" * 80)
    print("📋 GPU VALIDATION SUMMARY")
    print("=" * 80)
    print("\n🖥️  Hardware Information:")
    print(f"   GPU: {gpu_info['name']}")
    print(f"   VRAM: {gpu_info['memory_total']}MB total")
    print(f"   Driver: {gpu_info['driver_version']}")
    print(f"   CUDA: {gpu_info['cuda_version'] or 'Not available'}")
    print("\n📊 Test Results:")
    for test_name, result in test_results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {test_name.replace('_', ' ').title()}")
    if total_failures == 0 and all(test_results.values()):
        print("\n🎉 All GPU tests passed!")
        print("✅ GPU is fully functional for DocMind AI")
        _print_performance_notes(final_memory)
        sys.exit(0)
    print("\n❌ GPU validation issues detected")
    if total_failures > 0:
        print(f"   {total_failures} test suite(s) failed")
    print("\n💡 Recommendations:")
    if not test_results.get("cuda", True):
        print("   🔧 Install/update CUDA drivers")
    if not test_results.get("gpu_tests", True):
        print("   🎯 Check GPU-specific test failures")
    if not test_results.get("performance", True):
        print("   Investigate performance test issues with real models")
    sys.exit(1)


def _print_performance_notes(final_memory: dict[str, float] | None) -> None:
    """Emit performance guidance based on VRAM utilization."""
    print("\n💡 Performance Notes:")
    utilization = final_memory.get("utilization", 0) if final_memory else 0
    if utilization > 90:
        print(
            f"   ⚠️  High VRAM utilization ({utilization:.1f}%) - "
            "consider model optimization"
        )
    elif utilization > 70:
        print(f"   ✅ Good VRAM utilization ({utilization:.1f}%)")
    else:
        print(f"   💡 Low VRAM utilization ({utilization:.1f}%) - GPU underutilized")


def main() -> None:
    """Main GPU validation runner."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    _print_header()

    exit_codes: list[int] = []
    test_results: dict[str, bool] = {}

    gpu_info = _require_gpu_info(cwd=project_root)
    cuda_available = _check_cuda_compatibility_status(test_results, cwd=project_root)

    if args.compatibility:
        _print_compatibility_summary_and_exit(gpu_info, test_results)

    if not cuda_available:
        sys.exit(1)

    final_memory: dict[str, float] | None
    if args.quick:
        final_memory = _run_quick_check(exit_codes, test_results, cwd=project_root)
    else:
        final_memory = _run_gpu_tests(exit_codes, test_results, cwd=project_root)

    if args.benchmark:
        _run_benchmarks(project_root, exit_codes, test_results)
    if args.memory_check:
        _run_memory_leak_check(test_results, cwd=project_root)

    total_failures = sum(1 for code in exit_codes if code != 0)
    _print_summary(gpu_info, test_results, total_failures, final_memory)


if __name__ == "__main__":
    main()
