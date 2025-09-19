"""Unit tests for ``src.eval.common.determinism`` helpers."""

from __future__ import annotations

import os
import random
import sys
from types import SimpleNamespace

import pytest

from src.eval.common.determinism import set_determinism


@pytest.mark.unit
def test_set_determinism_applies_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_SEED", raising=False)
    monkeypatch.delenv("EVAL_THREADS", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    set_determinism(seed=7, threads=3)

    assert os.environ["PYTHONHASHSEED"] == "7"
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert random.randint(0, 10) == 5  # deterministic sequence for seed 7


@pytest.mark.unit
def test_set_determinism_handles_optional_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, list] = {"torch": [], "numpy": []}

    def _record_numpy(value: int) -> None:
        calls["numpy"].append(value)

    dummy_numpy = SimpleNamespace(
        random=SimpleNamespace(seed=_record_numpy),
        seed=_record_numpy,
    )

    def _cuda_is_available() -> bool:
        return True

    def _cuda_manual_seed_all(value: int) -> None:
        calls.setdefault("cuda", []).append(value)

    dummy_cuda = SimpleNamespace(
        is_available=_cuda_is_available,
        manual_seed_all=_cuda_manual_seed_all,
    )

    dummy_cudnn = SimpleNamespace(deterministic=False, benchmark=True)

    def _torch_manual_seed(value: int) -> None:
        calls["torch"].append(value)

    def _torch_set_num_threads(value: int) -> None:
        calls.setdefault("threads", []).append(value)

    def _torch_use_deterministic(flag: bool) -> None:
        calls.setdefault("deterministic", []).append(flag)

    dummy_torch = SimpleNamespace(
        cuda=dummy_cuda,
        backends=SimpleNamespace(cudnn=dummy_cudnn),
        manual_seed=_torch_manual_seed,
        set_num_threads=_torch_set_num_threads,
        use_deterministic_algorithms=_torch_use_deterministic,
    )

    monkeypatch.setitem(sys.modules, "numpy", dummy_numpy)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    set_determinism(seed=11, threads=2)

    assert calls["torch"] == [11]
    assert calls["cuda"] == [11]
    assert calls["threads"] == [2]
    assert calls["deterministic"] == [True]
