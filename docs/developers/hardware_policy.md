# Hardware Device & VRAM Policy

## Overview

DocMind AI centralizes device selection and VRAM capability checks in `src/utils/core.py`.
All business logic must use these helpers, not direct `torch` calls, to keep behavior
consistent across the codebase.

## Core Helpers

- `select_device(prefer: str = "auto") -> str`
  - Returns `"cuda" | "mps" | "cpu"` based on availability and preference.
- `resolve_device(prefer: str = "auto") -> tuple[str, int | None]`
  - Returns canonical device string and device index when applicable, e.g. `("cuda:0", 0)`,
    `("mps", None)`, or `("cpu", None)`.
- `has_cuda_vram(min_gb: float, device_index: int = 0) -> bool`
  - Returns `True` if CUDA is available and the queried device’s total VRAM (in raw bytes)
    meets the threshold (GiB). Uses `settings.monitoring.bytes_to_gb_divisor`.
- `get_vram_gb(device_index: int = 0) -> float | None`
  - Reports total VRAM for display/telemetry, using safe divisor access and rounding for readability.

## Usage Guidelines

- Always call `resolve_device()` in multi-GPU scenarios and propagate `device_index` to `has_cuda_vram`.
- Prefer `select_device` / `resolve_device` over ad-hoc device checks.
- Do not use per-call device overrides for encoders; instantiate with the desired `device` instead.
- Support MPS: where models/tensors are moved to `"cuda"`, also add corresponding `"mps"` branch.

## Measurement vs Policy

- Policy checks: use the helpers above (business logic).
- Measurement-only utilities: may use direct `torch` calls (e.g., `torch.cuda.memory_allocated`)
  in tests or low-level probes, but must not make policy decisions.

## Examples

```python
from src.utils.core import resolve_device, has_cuda_vram

dev_str, dev_idx = resolve_device("auto")
if dev_str.startswith("cuda") and has_cuda_vram(16.0, device_index=int(dev_idx or 0)):
    # Enable GPU-heavy path
    pass
```

## Future Work (Flag-Gated)

- Allocator-aware VRAM policy using `torch.cuda.mem_get_info` or NVML.
- Optional TTL cache for VRAM checks.
- Opt-in telemetry for device resolution and VRAM gating decisions.
