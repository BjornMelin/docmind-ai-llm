"""Unit tests for ``src.eval.common.determinism`` helpers."""

from __future__ import annotations

import os
import random
import sys

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

    class DummyNumpy:
        class Random:
            @staticmethod
            def seed(value: int) -> None:
                calls["numpy"].append(value)

        random = Random()

        @staticmethod
        def seed(value: int) -> None:
            calls["numpy"].append(value)

    class DummyCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def manual_seed_all(value: int) -> None:
            calls.setdefault("cuda", []).append(value)

    class DummyBackends:
        class CudnnBackend:  # type: ignore[assignment]
            deterministic = False
            benchmark = True

        cudnn = CudnnBackend()

    class DummyTorch:
        cuda = DummyCuda()
        backends = DummyBackends()

        @staticmethod
        def manual_seed(value: int) -> None:
            calls["torch"].append(value)

        @staticmethod
        def set_num_threads(value: int) -> None:
            calls.setdefault("threads", []).append(value)

        @staticmethod
        def use_deterministic_algorithms(flag: bool) -> None:
            calls.setdefault("deterministic", []).append(flag)

    monkeypatch.setitem(sys.modules, "numpy", DummyNumpy())
    monkeypatch.setitem(sys.modules, "torch", DummyTorch())

    set_determinism(seed=11, threads=2)

    assert calls["torch"] == [11]
    assert calls["cuda"] == [11]
    assert calls["threads"] == [2]
    assert calls["deterministic"] == [True]
