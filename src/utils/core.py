"""Canonical device selection and CUDA capacity helpers."""

try:  # Optional torch - allow module import without GPU stack present
    import torch as TORCH  # type: ignore  # noqa: N812
except ImportError:  # pragma: no cover - torch may be unavailable in CI
    TORCH = None  # type: ignore[assignment]

BYTES_PER_GIB = 1024**3


def is_cuda_available() -> bool:
    """Return True when CUDA is available via torch."""
    try:
        cuda_mod = getattr(TORCH, "cuda", None)
        return bool(cuda_mod and cuda_mod.is_available())
    except (AttributeError, RuntimeError, TypeError):  # pragma: no cover - conservative
        return False


def _is_mps_available() -> bool:
    """Return True when Apple MPS backend is available."""
    try:
        backends = getattr(TORCH, "backends", None)
        mps = getattr(backends, "mps", None) if backends else None
        return bool(mps) and bool(getattr(mps, "is_available", lambda: False)())
    except (AttributeError, RuntimeError, TypeError):  # pragma: no cover - conservative
        return False


def get_vram_gb(device_index: int = 0) -> float | None:
    """Return total GPU VRAM in GiB for a CUDA device; else None.

    Args:
        device_index: CUDA device index to query (default: 0).
    """
    if not is_cuda_available():
        return None
    try:
        cuda_mod = getattr(TORCH, "cuda", None)
        if not cuda_mod:
            return None
        total_bytes = cuda_mod.get_device_properties(int(device_index)).total_memory
        return round(total_bytes / BYTES_PER_GIB, 1)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
    ):  # pragma: no cover - conservative
        return None


def resolve_device(prefer: str = "auto") -> tuple[str, int | None]:
    """Resolve a canonical device string and device index when applicable.

    Args:
        prefer: 'auto'|'cpu'|'mps'|'cuda' or a concrete 'cuda:N'.

    Returns:
        Tuple of (device_str, device_index or None for CPU/MPS).
    """
    try:
        p = (prefer or "auto").lower()
        # If explicit cuda:N provided
        if p.startswith("cuda:"):
            try:
                idx = int(p.split(":", 1)[1])
            except (ValueError, TypeError):
                idx = 0
            return (f"cuda:{idx}", idx)
        # Use existing selection logic
        dev = select_device(p)
        if dev == "cuda":
            # Derive an index; prefer current device
            try:
                cuda_mod = getattr(TORCH, "cuda", None)
                idx = int(cuda_mod.current_device()) if cuda_mod else 0
            except (RuntimeError, AttributeError, TypeError, ValueError):
                idx = 0
            return (f"cuda:{idx}", idx)
        if dev == "mps":
            return ("mps", None)
        return ("cpu", None)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return ("cpu", None)


def has_cuda_vram(min_gb: float, device_index: int = 0) -> bool:
    """Return True when CUDA is available and total VRAM ≥ ``min_gb`` for a device.

    Args:
        min_gb: Minimum required VRAM (GiB) to consider the device sufficient.
        device_index: CUDA device index to check (default: 0).

    Returns:
        bool: True when CUDA is available and device VRAM meets the threshold;
              False otherwise.
    """
    try:
        if not is_cuda_available():
            return False
        # Compare using raw bytes to avoid rounding edge cases
        cuda_mod = getattr(TORCH, "cuda", None)
        if not cuda_mod:
            return False
        props = cuda_mod.get_device_properties(int(device_index))
        total_bytes = float(getattr(props, "total_memory", 0.0))
        required_bytes = float(min_gb) * BYTES_PER_GIB
        return total_bytes >= required_bytes
    except (RuntimeError, AttributeError, ImportError, TypeError, ValueError):
        return False


def select_device(prefer: str = "auto") -> str:
    """Select an inference device string ('cuda'|'mps'|'cpu').

    Preference order:
    - When ``prefer`` is 'cpu' or 'cuda' or 'mps', honor if available; else fall back.
    - When ``prefer`` is 'auto', choose 'cuda' if available; else 'mps' (Apple Silicon);
      otherwise 'cpu'.

    Returns:
        str: One of 'cuda', 'mps', or 'cpu'.
    """
    p = (prefer or "auto").lower()
    selected = "cpu"
    try:
        if p == "cpu":
            selected = "cpu"
        else:
            cuda_ok = is_cuda_available()
            mps_ok = _is_mps_available()
            if p == "cuda":
                selected = "cuda" if cuda_ok else "cpu"
            elif p == "mps":
                selected = "mps" if mps_ok else ("cuda" if cuda_ok else "cpu")
            else:  # auto
                selected = "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")
    except (RuntimeError, AttributeError, ImportError, TypeError):
        selected = "cpu"
    return selected
