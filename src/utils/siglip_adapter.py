"""SigLIP visual embedding adapter for LlamaIndex-like usage.

Provides a minimal object exposing `get_image_embedding(image)` returning a
numpy vector, suitable for MultiModalVectorStoreIndex flows that expect a
CLIP-like interface.
"""

from typing import Any

import numpy as np
from loguru import logger


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
            except (ImportError, AttributeError):
                model_id = None
        self.model_id = model_id or "google/siglip-base-patch16-224"
        # Delegate device selection to shared helper with safe fallback
        self.device = device or self._choose_device()
        self._model: Any | None = None
        self._proc: Any | None = None
        self._dim: int | None = None

    def _choose_device(self) -> str:
        """Select device via core helper; fall back to CPU when torch import fails.

        Attempt a torch import to detect runtime availability, then delegate to
        the shared select_device() helper when present.
        """
        try:
            import torch  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            return "cpu"
        try:
            from src.utils.core import select_device as _sd

            return _sd("auto")
        except (ImportError, AttributeError):  # pragma: no cover - conservative
            # Robust fallback using torch directly
            try:
                if (
                    getattr(torch, "cuda", None) and torch.cuda.is_available()  # type: ignore[attr-defined]
                ):
                    return "cuda"
                mps = getattr(getattr(torch, "backends", None), "mps", None)
                if mps is not None and getattr(mps, "is_available", lambda: False)():
                    return "mps"
                return "cpu"
            except AttributeError:
                return "cpu"

    def _load_siglip_transformers(self) -> None:
        """Direct transformers-based SigLIP loading fallback."""
        from transformers import SiglipModel, SiglipProcessor  # type: ignore

        model: Any = SiglipModel.from_pretrained(self.model_id)
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
        except (ImportError, AttributeError):  # pragma: no cover - settings import edge
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
        except (AttributeError, ValueError, TypeError):
            self._dim = None

    def _move_to_device(self, tensor: Any | None) -> Any | None:
        if tensor is None:
            return None
        if self.device == "cuda":
            return tensor.to("cuda")
        if self.device == "mps":
            return tensor.to("mps")
        return tensor

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

            inputs = self._proc(images=[image], return_tensors="pt")  # type: ignore[call-arg]
            pix = inputs.get("pixel_values")
            if pix is None:  # pragma: no cover - defensive
                raise RuntimeError("SigLIP processor returned no pixel_values")
            pix = self._move_to_device(pix)
            with torch.no_grad():  # type: ignore[name-defined]
                feats = self._model.get_image_features(pixel_values=pix)  # type: ignore[union-attr]
                feats = feats / feats.norm(dim=-1, keepdim=True)
                vec = feats[0].detach().cpu().numpy().astype(np.float32)
            return vec
        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError):
            dim = int(self._dim or 768)
            return np.zeros(dim, dtype=np.float32)

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Return L2-normalized SigLIP text features for a single text query.

        This enables cross-modal text->image retrieval in the same SigLIP space.
        """
        self._ensure_loaded()
        if self._model is None or self._proc is None:
            dim = int(self._dim or 768)
            return np.zeros(dim, dtype=np.float32)
        try:
            import torch  # type: ignore

            inputs = self._proc(  # type: ignore[call-arg]
                text=[str(text)],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            input_ids = inputs.get("input_ids")
            attn = inputs.get("attention_mask")
            input_ids = self._move_to_device(input_ids)
            attn = self._move_to_device(attn)
            with torch.no_grad():  # type: ignore[name-defined]
                feats = self._model.get_text_features(  # type: ignore[union-attr]
                    input_ids=input_ids,
                    attention_mask=attn,
                )
                feats = feats / feats.norm(dim=-1, keepdim=True)
                vec = feats[0].detach().cpu().numpy().astype(np.float32)
            return vec
        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError):
            dim = int(self._dim or 768)
            return np.zeros(dim, dtype=np.float32)

    def get_image_embeddings(
        self, images: list[Any], *, batch_size: int = 8
    ) -> np.ndarray:
        """Return L2-normalized SigLIP image features for a batch of images."""
        if not images:
            dim = int(self._dim or 768)
            return np.empty((0, dim), dtype=np.float32)
        self._ensure_loaded()
        if self._model is None or self._proc is None:
            dim = int(self._dim or 768)
            return np.zeros((len(images), dim), dtype=np.float32)

        try:
            import torch  # type: ignore
        except (ImportError, ModuleNotFoundError):  # pragma: no cover
            dim = int(self._dim or 768)
            return np.zeros((len(images), dim), dtype=np.float32)

        bs = max(1, int(batch_size))
        out: list[np.ndarray] = []
        try:
            for i in range(0, len(images), bs):
                batch = images[i : i + bs]
                inputs = self._proc(images=batch, return_tensors="pt")  # type: ignore[call-arg]
                pix = inputs.get("pixel_values")
                if pix is None:  # pragma: no cover - defensive
                    raise RuntimeError("SigLIP processor returned no pixel_values")
                pix = self._move_to_device(pix)
                with torch.no_grad():  # type: ignore[name-defined]
                    feats = self._model.get_image_features(pixel_values=pix)  # type: ignore[union-attr]
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    out.append(feats.detach().cpu().numpy().astype(np.float32))
        except Exception as exc:
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(str(exc), key_id="siglip.images")
            logger.debug(
                "siglip image embedding failed (count={} dim={} "
                "error_type={} error={})",
                len(images),
                self._dim,
                type(exc).__name__,
                redaction.redacted,
            )
            dim = int(self._dim or 768)
            return np.zeros((len(images), dim), dtype=np.float32)
        return (
            np.concatenate(out, axis=0)
            if out
            else np.empty((0, int(self._dim or 768)), dtype=np.float32)
        )

    def get_text_embeddings(
        self, texts: list[str], *, batch_size: int = 8
    ) -> np.ndarray:
        """Return L2-normalized SigLIP text features for a batch of texts."""
        if not texts:
            dim = int(self._dim or 768)
            return np.empty((0, dim), dtype=np.float32)
        self._ensure_loaded()
        if self._model is None or self._proc is None:
            dim = int(self._dim or 768)
            return np.zeros((len(texts), dim), dtype=np.float32)

        try:
            import torch  # type: ignore
        except (ImportError, ModuleNotFoundError):  # pragma: no cover
            dim = int(self._dim or 768)
            return np.zeros((len(texts), dim), dtype=np.float32)

        bs = max(1, int(batch_size))
        out: list[np.ndarray] = []
        try:
            for i in range(0, len(texts), bs):
                batch = [str(t) for t in texts[i : i + bs]]
                inputs = self._proc(  # type: ignore[call-arg]
                    text=batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                )
                input_ids = inputs.get("input_ids")
                attn = inputs.get("attention_mask")
                input_ids = self._move_to_device(input_ids)
                attn = self._move_to_device(attn)
                with torch.no_grad():  # type: ignore[name-defined]
                    feats = self._model.get_text_features(  # type: ignore[union-attr]
                        input_ids=input_ids,
                        attention_mask=attn,
                    )
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    out.append(feats.detach().cpu().numpy().astype(np.float32))
        except Exception as exc:
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(str(exc), key_id="siglip.texts")
            logger.debug(
                "siglip text embedding failed (count={} dim={} error_type={} error={})",
                len(texts),
                self._dim,
                type(exc).__name__,
                redaction.redacted,
            )
            dim = int(self._dim or 768)
            return np.zeros((len(texts), dim), dtype=np.float32)
        return (
            np.concatenate(out, axis=0)
            if out
            else np.empty((0, int(self._dim or 768)), dtype=np.float32)
        )
