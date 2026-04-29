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
def _cached(model_id: str, revision: str | None, device: str) -> tuple[Any, Any, str]:
    from transformers import SiglipModel, SiglipProcessor  # type: ignore

    if revision is None:
        model: Any = SiglipModel.from_pretrained(model_id)  # nosec B615
    else:
        model = SiglipModel.from_pretrained(  # nosec B615
            model_id,
            revision=revision,
        )
    if device in {"cuda", "mps"} and hasattr(model, "to"):
        model = model.to(device)
    if revision is None:
        processor = SiglipProcessor.from_pretrained(model_id)  # nosec B615
    else:
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
    resolved_revision = _resolve_revision(resolved_id, revision)
    dev = select_device(device or "auto")
    return _cached(resolved_id, resolved_revision, dev)


def _resolve_revision(model_id: str, revision: str | None) -> str | None:
    """Resolve the revision pin without applying default pins to custom models."""
    if model_id == DEFAULT_SIGLIP_MODEL_ID:
        return revision or DEFAULT_SIGLIP_MODEL_REVISION
    if revision == DEFAULT_SIGLIP_MODEL_REVISION:
        return None
    return revision
