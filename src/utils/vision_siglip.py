"""Shared SigLIP loader utility (library-first, cached).

Provides a single, cached entry point to load a SigLIP model and its processor
with device selection delegated to utils.core.select_device.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.utils.core import select_device


@lru_cache(maxsize=16)
def _cached(model_id: str, device: str) -> tuple[Any, Any, str]:
    from transformers import SiglipModel, SiglipProcessor  # type: ignore

    model = SiglipModel.from_pretrained(model_id)
    if device == "cuda" and hasattr(model, "to"):
        model = model.to("cuda")
    processor = SiglipProcessor.from_pretrained(model_id)
    return model, processor, device


def load_siglip(
    model_id: str | None = None, device: str | None = None
) -> tuple[Any, Any, str]:
    """Load SigLIP model+processor and choose device.

    Args:
        model_id: Hugging Face model id. Defaults to google/siglip-base-patch16-224.
        device: Preferred device ("auto"|"cpu"|"cuda"|"mps"). Defaults to auto.

    Returns:
        (model, processor, device_str)
    """
    resolved_id = model_id or "google/siglip-base-patch16-224"
    dev = select_device(device or "auto")
    return _cached(resolved_id, dev)
