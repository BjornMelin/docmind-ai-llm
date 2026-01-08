"""Unified embedding models and light-weight embedders.

This module provides small, library-first helpers for the SPEC-003 embedding
stack while keeping imports lazy to preserve offline determinism and clean
startup semantics.

Contents:
- Pydantic models describing parameters/results for text embeddings.
- TextEmbedder using BGE-M3 (dense + sparse) via FlagEmbedding (lazy import).
- ImageEmbedder with tiered backbones (OpenCLIP/SigLIP; optional Visualized-BGE),
  loaded lazily and normalized outputs.
- UnifiedEmbedder that routes items to the appropriate embedder.

Notes:
- No downloads or heavy imports occur at import time. Backends are loaded on
  first use. Private/protected seams may be patched at runtime where needed to
  avoid external dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

try:  # Optional torch for normalization and device checks
    import torch as TORCH  # type: ignore  # noqa: N812
except ImportError:  # pragma: no cover - torch may be unavailable in CI
    TORCH = None  # type: ignore[assignment]


class EmbeddingParameters(BaseModel):
    """Configuration parameters for BGE-M3 embedding operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_length: int = Field(
        default=8192, ge=512, le=16384, description="Maximum token length (8K context)"
    )
    use_fp16: bool = Field(default=True, description="Enable FP16 acceleration")
    normalize_embeddings: bool = Field(
        default=True, description="L2 normalize embeddings"
    )
    return_dense: bool = Field(
        default=True, description="Return dense embeddings (1024D)"
    )
    return_sparse: bool = Field(default=True, description="Return sparse embeddings")
    return_colbert: bool = Field(
        default=False, description="Return ColBERT multi-vector embeddings"
    )
    device: str = Field(
        default="cuda",
        description="Target device (cuda/mps/cpu/auto or 'cuda:N')",
    )
    pooling_method: str = Field(
        default="cls", description="Pooling method ('cls', 'mean')"
    )
    weights_for_different_modes: list[float] = Field(
        default=[0.4, 0.2, 0.4],
        description="Weights for [dense, sparse, colbert] fusion",
    )
    return_numpy: bool = Field(
        default=False, description="Return numpy arrays instead of lists"
    )


class EmbeddingResult(BaseModel):
    """Result of BGE-M3 embedding operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dense_embeddings: list[list[float]] | None = Field(
        default=None, description="Dense embeddings (1024D)"
    )
    sparse_embeddings: list[dict[int, float]] | None = Field(
        default=None, description="Sparse embeddings"
    )
    colbert_embeddings: list[np.ndarray] | None = Field(
        default=None, description="ColBERT multi-vector embeddings"
    )
    processing_time: float = Field(description="Embedding processing time in seconds")
    batch_size: int = Field(description="Batch size used")
    memory_usage_mb: float = Field(description="Peak GPU memory usage in MB")
    model_info: dict[str, Any] = Field(
        default_factory=dict, description="Model information"
    )


# ===== Implementation helpers =====
def _l2_normalize(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize a numpy array, guarding zero-norm rows.

    Args:
        arr: Input array.
        axis: Axis to normalize over.

    Returns:
        Normalized array with the same shape/dtype.
    """
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    # Avoid division by zero: only divide where norm > 0
    safe = np.where(norm > 0, arr / norm, arr)
    return safe.astype(arr.dtype, copy=False)


def _select_device(explicit: str | None = None) -> str:
    """Select device via shared core helper; explicit overrides."""
    if explicit:
        return explicit
    try:
        from src.utils.core import select_device as _sd

        return _sd("auto")
    except (ImportError, AttributeError):  # pragma: no cover - conservative fallback
        # Robust fallback using local TORCH checks when core import fails
        try:
            cuda_mod = getattr(TORCH, "cuda", None)
            if TORCH is not None and cuda_mod and cuda_mod.is_available():
                return "cuda"
            mps = getattr(getattr(TORCH, "backends", None), "mps", None)
            if mps is not None and getattr(mps, "is_available", lambda: False)():
                return "mps"
            return "cpu"
        except (AttributeError, RuntimeError):  # pragma: no cover - final guard
            return "cpu"


# ===== Text: BGE-M3 =====
class TextEmbedder:
    """Text embedding via BGE-M3 with optional sparse output.

    Library-first usage of FlagEmbedding.BGEM3FlagModel, imported lazily only
    when needed. All outputs can be L2-normalized for deterministic behavior.
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        use_fp16: bool | None = None,
        default_batch_size: int = 12,
        seed: int = 42,
    ) -> None:
        """Create a TextEmbedder with lazy-loaded BGEM3 backend.

        Args:
            model_name: HF model id ("BAAI/bge-m3").
            device: Preferred device ("cpu"|"cuda"|None for auto).
            use_fp16: Optional FP16 toggle (defaults to True on CUDA).
            default_batch_size: Batch size when none is provided.
            seed: Seed for reproducible behavior when torch is present.
        """
        self.model_name = model_name
        self.device = _select_device(device)
        self.use_fp16 = (
            bool(use_fp16) if use_fp16 is not None else (self.device == "cuda")
        )
        self.default_batch_size = int(default_batch_size)
        self._backend: Any | None = None
        self.seed = int(seed)
        self._dense_dim: int | None = 1024  # default for BGE-M3

    # --- Backend loading ---
    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except (ImportError, RuntimeError, AttributeError) as exc:
            msg = (
                "FlagEmbedding is required for TextEmbedder. "
                "Install with: uv add FlagEmbedding"
            )
            raise ImportError(msg) from exc

        if TORCH is not None:
            from contextlib import suppress

            with suppress(Exception):  # determinism best-effort
                TORCH.manual_seed(self.seed)

        self._backend = BGEM3FlagModel(
            self.model_name, use_fp16=self.use_fp16, devices=[self.device]
        )
        # Best-effort inference of dense dimension
        try:  # pragma: no cover - relies on backend specifics
            out = self._backend.encode(
                ["a"],
                batch_size=1,
                max_length=16,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dv = out.get("dense_vecs")
            if dv is not None:
                arr = np.asarray(dv, dtype=np.float32)
                self._dense_dim = int(arr.shape[-1])
        except (RuntimeError, ValueError, TypeError):
            # Keep default
            self._dense_dim = self._dense_dim or 1024

    # --- Encoding ---
    def encode_text(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = True,
        batch_size: int | None = None,
        normalize: bool = True,
        device: str | None = None,
        max_length: int = 8192,
    ) -> dict[str, Any]:
        """Encode a batch of texts to dense and/or sparse representations.

        Returns a dictionary with optional keys: ``dense`` (np.ndarray [N,1024])
        and ``sparse`` (list[dict[int,float]]). The backend is invoked with
        the official flags for BGEM3 dense+sparse output.
        """
        if not texts:
            # Use inferred or default dense dimension without loading backend
            dense_dim = int(self._dense_dim or 1024)
            out: dict[str, Any] = {}
            if return_dense:
                out["dense"] = np.empty((0, dense_dim), dtype=np.float32)
            if return_sparse:
                out["sparse"] = []
            return out

        bs = int(batch_size or self.default_batch_size)
        # Do not mutate instance device during encode; per-call overrides are not
        # supported because the BGEM3 backend binds to the device at load time.
        if device and device != self.device:
            raise ValueError(
                "Per-call device override is not supported; "
                "instantiate TextEmbedder(device=...) instead."
            )

        # Load backend only after validating per-call arguments to avoid
        # unnecessary downloads or heavy imports in offline/unit environments.
        self._ensure_loaded()

        outputs = self._backend.encode(  # type: ignore[call-arg]
            texts,
            batch_size=bs,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

        result: dict[str, Any] = {}
        if (
            return_dense
            and "dense_vecs" in outputs
            and outputs["dense_vecs"] is not None
        ):
            dense = np.asarray(outputs["dense_vecs"], dtype=np.float32)
            if normalize and dense.size:
                dense = _l2_normalize(dense)
            result["dense"] = dense

        if (
            return_sparse
            and "lexical_weights" in outputs
            and outputs["lexical_weights"] is not None
        ):
            # Already a list[dict[int,float]]
            result["sparse"] = outputs["lexical_weights"]

        return result


# ===== Images: OpenCLIP / SigLIP (tiered) =====
_BackboneName = Literal[
    "auto",
    "openclip_vitl14",
    "openclip_vith14",
    "siglip_base",
    "bge_visualized",
]


class ImageEmbedder:
    """Tiered image embedder with lazy, library-first backends.

    Selection logic (heuristic):
    - CPU/low-VRAM → OpenCLIP ViT-L/14 (768D)
    - Mid-GPU → SigLIP base-patch16-224 (768D)
    - High-GPU → OpenCLIP ViT-H/14 (1024D)
    - Optional Visualized-BGE (off by default)
    """

    def __init__(
        self,
        *,
        backbone: _BackboneName = "auto",
        device: str | None = None,
        default_batch_size: int = 8,
        seed: int = 42,
    ) -> None:
        """Create an ImageEmbedder with lazy backbone selection.

        Args:
            backbone: One of
                "auto|openclip_vitl14|openclip_vith14|siglip_base|bge_visualized".
            device: Preferred device ("cpu"|"cuda"|None for auto).
            default_batch_size: Images per batch.
            seed: Reserved for future randomized transforms.
        """
        self.backbone = backbone
        self.device = _select_device(device)
        self.default_batch_size = int(default_batch_size)
        self.seed = int(seed)
        self._backend: Any | None = None  # model/module
        self._preprocess: Any | None = None  # callable for images → tensor
        self._dim: int | None = None

    # --- Backend loading helpers ---
    def _choose_auto_backbone(self) -> _BackboneName:
        # Centralized device/VRAM policy via src.utils.core
        if self.device == "cpu" or TORCH is None or not getattr(TORCH, "cuda", None):
            return "openclip_vitl14"
        try:
            from src.utils.core import (
                has_cuda_vram as _has_cuda_vram,
            )
            from src.utils.core import (
                resolve_device as _resdev,
            )

            # High-VRAM GPUs use OpenCLIP ViT-H/14; otherwise SigLIP base.
            _dev_str, _idx = _resdev(self.device)
            return (
                "openclip_vith14"
                if _has_cuda_vram(20.0, device_index=int(_idx or 0))
                else "siglip_base"
            )
        except (RuntimeError, ImportError, AttributeError, TypeError):
            # Fallback heuristic if core helpers unavailable
            try:  # pragma: no cover
                cuda_mod = getattr(TORCH, "cuda", None)
                if cuda_mod and cuda_mod.is_available():
                    props = cuda_mod.get_device_properties(0)
                    total = float(props.total_memory) / float(1024**3)
                    return "openclip_vith14" if total >= 20.0 else "siglip_base"
            except (RuntimeError, AttributeError):
                return "openclip_vitl14"
            return "siglip_base"

    def _load_openclip(self, model_name: str) -> None:
        import importlib

        open_clip = importlib.import_module("open_clip")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="laion2b_s34b_b79k"
        )
        model.eval()
        if self.device == "cuda":  # move if needed
            model = model.to("cuda")
        elif self.device == "mps":
            model = model.to("mps")
        self._backend = model
        self._preprocess = preprocess
        # Derive output dimension from model when possible
        dim: int | None = None
        try:
            visual = getattr(model, "visual", None)
            if visual is not None:
                dim = int(getattr(visual, "output_dim", 0)) or None
            if dim is None and hasattr(model, "embed_dim"):
                dim = int(model.embed_dim)  # type: ignore[arg-type]
        except (AttributeError, ValueError, TypeError):  # pragma: no cover
            dim = None

        self._dim = dim or (1024 if ("H-14" in model_name) else 768)

    def _load_siglip(self) -> None:
        try:
            from src.utils.vision_siglip import load_siglip

            try:
                from src.config import settings as _settings

                model_id = getattr(
                    _settings.embedding,
                    "siglip_model_id",
                    "google/siglip-base-patch16-224",
                )
            except (AttributeError, ImportError, RuntimeError, ValueError, TypeError):
                model_id = "google/siglip-base-patch16-224"
            model, processor, _dev = load_siglip(model_id=model_id, device=self.device)
            self._backend = model
            self._preprocess = processor
        except (ImportError, RuntimeError, AttributeError, ValueError, TypeError):
            self._load_siglip_transformers_fallback()

    def _load_siglip_transformers_fallback(self) -> None:
        """Direct transformers-based SigLIP loading as a fallback path."""
        from transformers import SiglipModel, SiglipProcessor  # type: ignore

        model_id = "google/siglip-base-patch16-224"
        model = SiglipModel.from_pretrained(model_id, device_map=None)
        if self.device in ("cuda", "mps"):
            model = cast(Any, model).to(self.device)
        processor = SiglipProcessor.from_pretrained(model_id)
        self._backend = model
        self._preprocess = processor

    def _load_bge_visualized(self) -> None:
        # Optional path; raise clear error if selected without deps
        raise RuntimeError(
            "bge_visualized backbone is disabled by default. "
            "Enable explicitly with proper GPUs and dependencies."
        )

    def _ensure_loaded(self, override: _BackboneName | None = None) -> None:
        if self._backend is not None:
            return
        name = override or self.backbone or "auto"
        if name == "auto":
            name = self._choose_auto_backbone()

        if name == "openclip_vitl14":
            self._load_openclip("ViT-L-14")
        elif name == "openclip_vith14":
            self._load_openclip("ViT-H-14")
        elif name == "siglip_base":
            self._load_siglip()
        elif name == "bge_visualized":
            self._load_bge_visualized()
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown backbone: {name}")

    # --- Encoding ---
    def encode_image(
        self,
        images: list[Any],  # PIL.Image.Image or array-like
        *,
        backbone: _BackboneName | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
        device: str | None = None,
    ) -> np.ndarray:
        """Encode images and return L2-normalized features.

        Args:
            images: List of PIL images (or compatible). Empty list returns shape (0,d).
            backbone: Optional backbone override for this call.
            batch_size: Optional batch size; defaults to constructor value.
            normalize: Whether to L2-normalize outputs.
            device: Optional device override.
        """
        if not images:
            # For empty input, avoid heavy loading when possible
            dim = int(self._dim or (1024 if (backbone == "openclip_vith14") else 768))
            return np.empty((0, dim), dtype=np.float32)

        # Disallow per-call device overrides to avoid device mismatch with loaded model
        if device and device != self.device:
            raise ValueError(
                "Per-call device override is not supported; "
                "instantiate ImageEmbedder(device=...) instead."
            )
        self._ensure_loaded(backbone)

        bs = int(batch_size or self.default_batch_size)
        feats: list[np.ndarray] = []

        backend = self._backend
        preprocess = self._preprocess

        # OpenCLIP path exposes a torchvision-like preprocess
        if (
            backend is not None
            and preprocess is not None
            and callable(preprocess)
            and hasattr(backend, "encode_image")
        ):
            backend_any = cast(Any, backend)
            for i in range(0, len(images), bs):
                batch = images[i : i + bs]
                tensors = [cast(Any, preprocess)(img) for img in batch]
                # Stack to [B, C, H, W]
                if TORCH is None:  # pragma: no cover - explicit failure in prod
                    raise RuntimeError(
                        "OpenCLIP image feature extraction requires torch."
                    )
                x = TORCH.stack(tensors)  # type: ignore[arg-type]
                if self.device == "cuda":
                    x = x.to("cuda")
                elif self.device == "mps":
                    x = x.to("mps")
                with TORCH.no_grad():
                    f = backend_any.encode_image(x)
                    f = f / f.norm(dim=-1, keepdim=True) if normalize else f
                    feats.append(f.detach().cpu().numpy().astype(np.float32))

            return (
                np.concatenate(feats, axis=0)
                if feats
                else np.empty((0, int(self._dim or 768)), dtype=np.float32)
            )

        # SigLIP path via HF processors
        if (
            backend is not None
            and preprocess is not None
            and hasattr(backend, "get_image_features")
        ):
            backend_any = cast(Any, backend)
            # Processor returns dict with pixel_values
            proc = cast(Any, preprocess)
            out_list: list[np.ndarray] = []
            for i in range(0, len(images), bs):
                batch = images[i : i + bs]
                if TORCH is None:  # pragma: no cover
                    raise RuntimeError(
                        "SigLIP image feature extraction requires torch."
                    )
                inputs = cast(dict[str, Any], proc(images=batch, return_tensors="pt"))
                pix = inputs.get("pixel_values")
                if pix is None:  # pragma: no cover - defensive
                    raise RuntimeError("SigLIP processor returned no pixel_values")
                if self.device == "cuda":
                    pix = pix.to("cuda")
                elif self.device == "mps":
                    pix = pix.to("mps")
                with TORCH.no_grad():
                    f = backend_any.get_image_features(pixel_values=pix)
                    if normalize:
                        f = f / f.norm(dim=-1, keepdim=True)
                    out_list.append(f.detach().cpu().numpy().astype(np.float32))
            return np.concatenate(out_list, axis=0)

        raise RuntimeError("Image backend not initialized correctly")


class UnifiedEmbedder:
    """Simple router that delegates to TextEmbedder and ImageEmbedder."""

    def __init__(
        self,
        *,
        text: TextEmbedder | None = None,
        image: ImageEmbedder | None = None,
        strict_image_types: bool = False,
    ) -> None:
        """Initialize with optional custom sub-embedders."""
        self.text = text or TextEmbedder()
        self.image = image or ImageEmbedder()
        self.strict_image_types = bool(strict_image_types)

    def encode(self, items: list[str | Any]) -> dict[str, Any]:
        """Encode a mixed list of strings and images via routing, with type checks."""
        texts: list[str] = []
        imgs: list[Any] = []
        for it in items:
            if isinstance(it, str):
                texts.append(it)
            else:
                if self.strict_image_types:
                    try:
                        from PIL import Image as _PILImage  # type: ignore

                        pil_type = _PILImage.Image
                    except (ImportError, AttributeError):  # pragma: no cover - optional
                        pil_type = None  # type: ignore[assignment]

                    supported = (np.ndarray,) + (
                        (pil_type,) if pil_type is not None else ()
                    )
                    # pylint: disable=isinstance-second-argument-not-valid-type
                    if not isinstance(it, supported):
                        t = type(it)
                        msg = f"Unsupported image type: {t}. Supported: {supported}"
                        raise TypeError(msg)
                imgs.append(it)

        out: dict[str, Any] = {}
        if texts:
            out |= self.text.encode_text(texts)
        if imgs:
            out["image_dense"] = self.image.encode_image(imgs)
        return out

    def encode_pair(
        self, texts: Iterable[str], images: Iterable[Any]
    ) -> dict[str, Any]:
        """Encode separate lists of texts and images in one call."""
        res: dict[str, Any] = {}
        t_list = list(texts)
        i_list = list(images)
        if t_list:
            res |= self.text.encode_text(t_list)
        if i_list:
            res["image_dense"] = self.image.encode_image(i_list)
        return res
