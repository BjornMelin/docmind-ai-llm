"""Shared SigLIP loader utility (library-first, cached).

Provides a single, cached entry point to load a SigLIP model and its processor
with device selection delegated to utils.core.select_device.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.utils.core import select_device

DEFAULT_SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
DEFAULT_SIGLIP_MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"


def siglip_features(output: Any, *, normalize: bool = True) -> Any:
    """Return SigLIP feature tensors across Transformers v4 and v5 APIs.

    Transformers v4 returned tensors directly from ``get_*_features``. In
    Transformers v5 those methods return model output objects with
    ``pooler_output``. Keeping this normalization local prevents each caller
    from depending on a specific Transformers major version.
    """
    features = getattr(output, "pooler_output", output)
    if normalize:
        return features / features.norm(dim=-1, keepdim=True)
    return features


@lru_cache(maxsize=16)
def _cached(model_id: str, revision: str, device: str) -> tuple[Any, Any, str]:
    from transformers import SiglipModel, SiglipProcessor  # type: ignore

    model: Any = SiglipModel.from_pretrained(  # nosec B615
        model_id,
        revision=revision,
    )
    if device in {"cuda", "mps"} and hasattr(model, "to"):
        model = model.to(device)
    processor = SiglipProcessor.from_pretrained(  # nosec B615
        model_id,
        revision=revision,
    )
    return model, processor, device


def load_siglip(
    model_id: str | None = None,
    device: str | None = None,
    revision: str | None = None,
) -> tuple[Any, Any, str]:
    """Load SigLIP model+processor and choose device.

    Args:
        model_id: Hugging Face model id. Defaults to google/siglip-base-patch16-224.
        device: Preferred device ("auto"|"cpu"|"cuda"|"mps"). Defaults to auto.
        revision: Pinned Hugging Face revision for supply-chain stability.

    Returns:
        (model, processor, device_str)
    """
    resolved_id = model_id or DEFAULT_SIGLIP_MODEL_ID
    resolved_revision = revision or DEFAULT_SIGLIP_MODEL_REVISION
    dev = select_device(device or "auto")
    return _cached(resolved_id, resolved_revision, dev)
