"""Regression tests for the GPU validation command boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import test_gpu


def test_gpu_info_uses_supported_nvidia_query(monkeypatch, tmp_path: Path) -> None:
    """Hardware discovery must only request fields supported by nvidia-smi."""
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="NVIDIA Test GPU, 16376, 16000, 595.97\n",
            stderr="",
        )

    monkeypatch.setattr(test_gpu.subprocess, "run", fake_run)

    assert test_gpu.get_gpu_info(cwd=tmp_path) == {
        "name": "NVIDIA Test GPU",
        "memory_total": 16376,
        "memory_free": 16000,
        "driver_version": "595.97",
    }
    assert commands[0][1] == (
        "--query-gpu=name,memory.total,memory.free,driver_version"
    )


def test_cuda_probe_uses_active_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The CUDA probe must not let uv replace the installed GPU profile."""
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="CUDA available: True\n",
            stderr="",
        )

    monkeypatch.setattr(test_gpu.subprocess, "run", fake_run)

    assert test_gpu.check_cuda_availability(cwd=tmp_path)
    assert commands == [
        [
            sys.executable,
            "-c",
            "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); "
            "print(f'CUDA devices: {torch.cuda.device_count()}'); "
            "current_dev = torch.cuda.current_device() if "
            "torch.cuda.is_available() else 'N/A'; "
            "print(f'Current device: {current_dev}'); "
            "dev_name = torch.cuda.get_device_name(0) if "
            "torch.cuda.is_available() else 'N/A'; "
            "print(f'Device name: {dev_name}')",
        ]
    ]


def test_gpu_pytest_commands_use_active_environment(monkeypatch) -> None:
    """Nested Pytest runs must preserve the caller-selected dependency profile."""
    commands: list[list[str]] = []

    def fake_run_command(
        command: list[str],
        _description: str,
        timeout: int = 1800,
        *,
        cwd: Path | None = None,
    ) -> tuple[int, str]:
        del timeout, cwd
        commands.append(command)
        return 0, ""

    monkeypatch.setattr(test_gpu, "run_command", fake_run_command)
    monkeypatch.setattr(
        test_gpu,
        "monitor_gpu_memory",
        lambda **_kwargs: {"used": 0.0, "total": 1.0, "free": 1.0, "utilization": 0.0},
    )

    test_gpu._run_quick_check([], {})
    test_gpu._run_gpu_tests([], {})

    assert [command[:3] for command in commands] == [
        [sys.executable, "-m", "pytest"],
        [sys.executable, "-m", "pytest"],
    ]
