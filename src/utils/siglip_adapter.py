"""SigLIP visual embedding adapter for LlamaIndex-like usage.

Provides a minimal object exposing `get_image_embedding(image)` returning a
numpy vector, suitable for MultiModalVectorStoreIndex flows that expect a
CLIP-like interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SiglipEmbedding:
    """Minimal SigLIP adapter exposing `get_image_embedding`.

    Loads model and processor lazily on first use. Returns L2-normalized
    numpy vectors.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize adapter with model ID and device.

        Args:
            model_id: Hugging Face model identifier for SigLIP.
            device: Optional device override ("cpu"|"cuda"). When None, auto.
        """
        if model_id is None:
            try:
                from src.config import settings as app_settings  # local import

                model_id = getattr(app_settings.embedding, "siglip_model_id", None)
            except Exception:  # pylint: disable=broad-exception-caught
                model_id = None
        self.model_id = model_id or "google/siglip-base-patch16-224"
        # Delegate device selection to shared helper with safe fallback
        self.device = device or self._choose_device()
        self._model: Any | None = None
        self._proc: Any | None = None
        self._dim: int | None = None

    def _choose_device(self) -> str:
        """Select device via core helper; fall back to CPU when torch import fails.

        This avoids test-only branches; we simply attempt an import to determine
        if the runtime has torch available and then defer to select_device.
        """
        try:
            import torch  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            return "cpu"
        try:
            from src.utils.core import select_device as _sd

            return _sd("auto")
        except Exception:  # pragma: no cover - conservative
            # Robust fallback using torch directly
            try:
                if getattr(torch, "cuda", None) and torch.cuda.is_available():  # type: ignore[attr-defined]
                    return "cuda"
                mps = getattr(getattr(torch, "backends", None), "mps", None)
                if mps is not None and getattr(mps, "is_available", lambda: False)():
                    return "mps"
                return "cpu"
            except Exception:
                return "cpu"

    def _load_siglip_transformers(self) -> None:
        """Direct transformers-based SigLIP loading as a compatibility path."""
        from transformers import SiglipModel, SiglipProcessor  # type: ignore

        model = SiglipModel.from_pretrained(self.model_id)
        if self.device in ("cuda", "mps"):
            model = model.to(self.device)
        proc = SiglipProcessor.from_pretrained(self.model_id)
        self._model = model
        self._proc = proc

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._proc is not None:
            return
        # Gate unified loader via settings flag
        use_unified = True
        try:
            from src.config import settings as app_settings

            use_unified = bool(
                getattr(app_settings.retrieval, "siglip_adapter_unified", True)
            )
        except Exception:  # pragma: no cover - settings import edge
            use_unified = True

        if use_unified:
            try:
                from src.utils.vision_siglip import load_siglip

                model, proc, dev = load_siglip(self.model_id, self.device)
                self.device = dev
                self._model = model
                self._proc = proc
            except (
                ImportError,
                AttributeError,
                RuntimeError,
                ValueError,
                TypeError,
            ):
                # Fallback to direct transformers path
                self._load_siglip_transformers()
        else:
            self._load_siglip_transformers()
        # best-effort dimension
        try:
            cfg = getattr(getattr(self, "_model", None), "config", None)
            proj = int(getattr(cfg, "projection_dim", 0)) if cfg is not None else 0
            self._dim = proj or None
        except Exception:  # pylint: disable=broad-exception-caught
            self._dim = None

    def get_image_embedding(self, image: Any) -> np.ndarray:
        """Return L2-normalized SigLIP features for a single image.

        Args:
            image: PIL Image or input accepted by SiglipProcessor.

        Returns:
            numpy.ndarray: 1-D vector of visual features.
        """
        self._ensure_loaded()
        if self._model is None or self._proc is None:
            dim = int(self._dim or 768)
            return np.zeros(dim, dtype=np.float32)
        try:
            import torch  # type: ignore

            inputs = self._proc(
                images=[image],
                return_tensors="pt",  # type: ignore[attr-defined]
            )
            pix = inputs.get("pixel_values")
            if self.device == "cuda":
                pix = pix.to("cuda")
            elif self.device == "mps":
                pix = pix.to("mps")
            with torch.no_grad():  # type: ignore[name-defined]
                feats = self._model.get_image_features(pixel_values=pix)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                vec = feats[0].detach().cpu().numpy().astype(np.float32)
            return vec
        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError):
            # Return zeros if inference fails unexpectedly
            dim = int(self._dim or 768)
            return np.zeros(dim, dtype=np.float32)
