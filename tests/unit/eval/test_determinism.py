from __future__ import annotations

import os
import random

import pytest

from src.eval.common.determinism import set_determinism


@pytest.mark.unit
def test_set_determinism_env_and_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    set_determinism(seed=123, threads=2)
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert os.environ["OMP_NUM_THREADS"] == "2"
    # RNG reproducibility (basic sanity)
    set_determinism(seed=5, threads=1)
    a = [random.randint(0, 1000) for _ in range(3)]
    set_determinism(seed=5, threads=1)
    b = [random.randint(0, 1000) for _ in range(3)]
    assert a == b
