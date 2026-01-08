"""Determinism helpers for the evaluation harness.

This module provides a single entry-point ``set_determinism`` that configures
Python, NumPy, and (optionally) PyTorch RNGs, and caps threads to achieve
reproducible behavior across platforms.

Environment variables respected:
    - ``EVAL_SEED``: overrides the RNG seed (default: 42)
    - ``EVAL_THREADS``: limits threads used by BLAS/NumExpr/Torch (default: 1)

Note:
    PyTorch is optional. If present, we set deterministic and threads flags in
    a best-effort way without failing if unavailable.
"""

from __future__ import annotations

import os
import random
from contextlib import suppress


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def set_determinism(seed: int | None = None, threads: int | None = None) -> None:
    """Apply deterministic settings for eval runs.

    Args:
        seed: RNG seed. Defaults to ``int(os.environ['EVAL_SEED'] or 42)``.
        threads: Thread cap. Defaults to ``int(os.environ['EVAL_THREADS'] or 1)``.
    """
    seed = int(seed if seed is not None else _int_env("EVAL_SEED", 42))
    threads = int(threads if threads is not None else _int_env("EVAL_THREADS", 1))

    # Python hashing and RNG
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    # NumPy (optional import)
    with suppress(Exception):  # pragma: no cover - optional at runtime
        import numpy as np  # type: ignore

        np.random.seed(seed)

    # PyTorch (optional import)
    with suppress(Exception):  # pragma: no cover - optional at runtime
        import torch  # type: ignore

        with suppress(Exception):
            torch.manual_seed(seed)
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        # Threads and determinism hints
        with suppress(Exception):
            torch.set_num_threads(threads)
        with suppress(Exception):
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        with suppress(Exception):
            from torch import backends as _b  # type: ignore

            if hasattr(_b, "cudnn"):
                _b.cudnn.deterministic = True  # type: ignore[attr-defined]
                _b.cudnn.benchmark = False  # type: ignore[attr-defined]

    # Thread env caps for BLAS libraries & tokenizers
    #
    # Set these explicitly (not via setdefault) so callers can reliably override
    # existing process-level defaults when they request deterministic behavior.
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


__all__ = ["set_determinism"]
