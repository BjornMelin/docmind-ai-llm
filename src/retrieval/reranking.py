"""Modality-aware reranking utilities per ADR-037 and SPEC-005.

Changes in this patch:
- Default visual rerank uses SigLIP text-image cosine.
- Optional ColPali auto-enables via policy thresholds (VRAM/budget/K/visual
  fraction).
- Rank-level RRF merge across modalities; fail-open on timeout.
"""

from __future__ import annotations

import time
from functools import cache
from typing import Any

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.postprocessor.colpali_rerank import ColPaliRerank
from loguru import logger

from src.config import settings
from src.utils.multimodal import TEXT_TRUNCATION_LIMIT

# Time budgets (ms)
TEXT_RERANK_TIMEOUT_MS = 250
SIGLIP_TIMEOUT_MS = 150
COLPALI_TIMEOUT_MS = 400

# Policy thresholds
COLPALI_MIN_VRAM_GB = 8.0  # enable when >= 8-12 GB
COLPALI_TOPK_MAX = 16  # small-K scenarios
SIGLIP_PRUNE_M = 64  # cascade: SigLIP prune to m → ColPali m'
COLPALI_FINAL_M = 16


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _has_cuda_vram(min_gb: float) -> bool:
    try:
        import torch

        if not torch.cuda.is_available():  # type: ignore[attr-defined]
            return False
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # type: ignore[attr-defined]
        return total >= float(min_gb)
    except Exception:
        return False


def _rrf_merge(
    lists: list[list[NodeWithScore]], k_constant: int
) -> list[NodeWithScore]:
    """Rank-level Reciprocal Rank Fusion over multiple reranked lists."""
    scores: dict[str, tuple[float, NodeWithScore]] = {}
    for ranked in lists:
        for rank, nws in enumerate(ranked, start=1):
            nid = nws.node.node_id
            inc = 1.0 / (k_constant + rank)
            cur = scores.get(nid)
            if cur is None:
                scores[nid] = (inc, nws)
            else:
                scores[nid] = (cur[0] + inc, cur[1])
    fused = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [n for _score, n in fused]


def _load_siglip() -> tuple[Any, Any, str]:  # (model, processor, device)
    """Lazy-load SigLIP model+processor and choose device.

    Returns: (model, processor, device_str)
    """
    from transformers import SiglipModel, SiglipProcessor  # type: ignore

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():  # type: ignore[attr-defined]
            device = "cuda"
    except Exception:
        device = "cpu"

    model_id = "google/siglip-base-patch16-224"
    model = SiglipModel.from_pretrained(model_id)
    if device == "cuda":
        model = model.to("cuda")
    processor = SiglipProcessor.from_pretrained(model_id)
    return model, processor, device


def _siglip_rescore(
    query: str, nodes: list[NodeWithScore], budget_ms: int
) -> list[NodeWithScore]:
    """Compute SigLIP text-image cosine scores for visual nodes.

    Returns nodes with updated ``score`` sorted desc, within time budget.
    Fails open (returns input order) on errors.
    """
    if not nodes:
        return nodes
    t0 = _now_ms()
    try:
        import torch  # local import to avoid global dependency in tests

        # Gather image paths
        paths: list[str] = []
        for n in nodes:
            meta = getattr(n.node, "metadata", {}) or {}
            p = meta.get("image_path") or meta.get("path")
            if not p:
                # Skip nodes without path; keep ordering later
                paths.append("")
            else:
                paths.append(str(p))

        # Load images lazily; stop if time runs out
        from PIL import Image  # type: ignore

        images: list[Any] = []
        for p in paths:
            if p:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    images.append(None)
            else:
                images.append(None)
            if _now_ms() - t0 > budget_ms:
                logger.warning("SigLIP load images timeout; fail-open")
                return nodes

        model, processor, device = _load_siglip()

        # Text features (1, D)
        txt_inputs = processor(text=[query], return_tensors="pt")
        if device == "cuda":
            for k, v in txt_inputs.items():
                txt_inputs[k] = v.to("cuda")
        with torch.no_grad():  # type: ignore[name-defined]
            tfeat = model.get_text_features(**txt_inputs)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        # Batched image features
        feats: list[float] = []
        for img in images:
            if _now_ms() - t0 > budget_ms:
                logger.warning("SigLIP compute timeout; fail-open")
                return nodes
            if img is None:
                feats.append(float("-inf"))
                continue
            im_inputs = processor(images=[img], return_tensors="pt")
            if device == "cuda":
                im_inputs["pixel_values"] = im_inputs["pixel_values"].to("cuda")
            with torch.no_grad():  # type: ignore[name-defined]
                if hasattr(model, "get_image_features"):
                    if device == "cuda":
                        # mypy: ensure proper device placement
                        pass
                    imfeat = model.get_image_features(**im_inputs)
                    imfeat = imfeat / imfeat.norm(dim=-1, keepdim=True)
                    # Cosine is dot after normalization
                    score = float((tfeat @ imfeat.T).squeeze().detach().cpu().numpy())
                else:
                    score = float("-inf")
            feats.append(score)

        # Update scores and sort
        for n, s in zip(nodes, feats, strict=False):
            n.score = s
            # Truncate text for visibility if needed (no UI dependency here)
            if (
                getattr(n.node, "text", None)
                and len(n.node.text) > TEXT_TRUNCATION_LIMIT
            ):
                n.node.text = n.node.text[:TEXT_TRUNCATION_LIMIT]
        nodes_sorted = sorted(nodes, key=lambda x: x.score or 0.0, reverse=True)
        return nodes_sorted[: settings.retrieval.reranking_top_k]
    except Exception as exc:
        logger.warning("SigLIP rerank error: {} — fail-open", exc)
        return nodes


def _parse_top_k(value: int | str | None) -> int:
    """Validate and convert ``top_n`` values to ``int``."""
    if value is None:
        return settings.retrieval.reranking_top_k
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid top_n: {value}") from exc


@cache
def _build_text_reranker_cached(top_n: int) -> SentenceTransformerRerank:
    return SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=top_n,
        use_fp16=True,
        normalize=settings.retrieval.reranker_normalize_scores,
    )


def build_text_reranker(top_n: int | str | None = None) -> SentenceTransformerRerank:
    """Create text CrossEncoder reranker (BGE v2-m3)."""
    k = _parse_top_k(top_n)
    return _build_text_reranker_cached(k)


@cache
def _build_visual_reranker_cached(top_n: int) -> ColPaliRerank:
    return ColPaliRerank(model="vidore/colpali-v1.2", top_n=top_n)


def build_visual_reranker(top_n: int | str | None = None) -> ColPaliRerank:
    """Create visual reranker (ColPali)."""
    k = _parse_top_k(top_n)
    try:
        return _build_visual_reranker_cached(k)
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - library quirk
        raise ValueError(f"ColPaliRerank initialization failed: {exc}") from exc


class MultimodalReranker(BaseNodePostprocessor):
    """Postprocessor that applies text and visual rerankers per node modality."""

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        if not nodes or not query_bundle:
            return nodes

        text_nodes, visual_nodes = self._split_by_modality(nodes)

        # Stage 2: text rerank with timeout
        lists: list[list[NodeWithScore]] = []
        t_start = _now_ms()
        try:
            if text_nodes:
                tr = build_text_reranker().postprocess_nodes(
                    text_nodes, query_str=query_bundle.query_str
                )
                lists.append(tr)
        except Exception as exc:
            logger.warning("Text rerank error: {} — fail-open", exc)

        if _now_ms() - t_start > TEXT_RERANK_TIMEOUT_MS:
            logger.warning("Text rerank timeout; fail-open")
            return nodes

        # Stage 3: visual rerank — SigLIP default within budget
        v_start = _now_ms()
        try:
            if visual_nodes:
                sr = _siglip_rescore(
                    query_bundle.query_str, visual_nodes, SIGLIP_TIMEOUT_MS
                )
                lists.append(sr)
        except Exception as exc:
            logger.warning("SigLIP rerank error: {} — continue without", exc)

        # Optional Stage 3b: ColPali policy
        try:
            if self._should_enable_colpali(visual_nodes, lists):
                # Cascade: SigLIP prune to m → ColPali final on m'
                base = lists[-1] if lists else visual_nodes
                pruned = base[:SIGLIP_PRUNE_M]
                cr = build_visual_reranker(
                    top_n=min(COLPALI_FINAL_M, settings.retrieval.reranking_top_k)
                ).postprocess_nodes(
                    pruned, query_str=query_bundle.query_str
                )
                lists.append(cr)
        except Exception as exc:
            logger.warning("ColPali rerank error: {} — continue without", exc)

        if _now_ms() - v_start > (SIGLIP_TIMEOUT_MS + COLPALI_TIMEOUT_MS):
            logger.warning("Visual rerank timeout; fail-open")
            return nodes

        if not lists:
            return nodes[: settings.retrieval.reranking_top_k]
        fused = _rrf_merge(
            lists, k_constant=int(getattr(settings.retrieval, "rrf_k_constant", 60))
        )
        # De-dup and cap
        seen: set[str] = set()
        out: list[NodeWithScore] = []
        for n in fused:
            if n.node.node_id in seen:
                continue
            seen.add(n.node.node_id)
            out.append(n)
            if len(out) >= settings.retrieval.reranking_top_k:
                break
        return out

    @staticmethod
    def _split_by_modality(
        nodes: list[NodeWithScore],
    ) -> tuple[list[NodeWithScore], list[NodeWithScore]]:
        text = [n for n in nodes if n.node.metadata.get("modality", "text") == "text"]
        visual = [
            n
            for n in nodes
            if n.node.metadata.get("modality") in {"image", "pdf_page_image"}
        ]
        return text, visual

    @staticmethod
    def _should_enable_colpali(
        visual_nodes: list[NodeWithScore],
        lists: list[list[NodeWithScore]],
    ) -> bool:
        """Activation heuristic for ColPali (policy thresholds).

        Enables when ALL apply:
        - visual_fraction high OR corpus flagged visual-heavy,
        - reranking_top_k ≤ 10-16,
        - GPU VRAM ≥ 8-12 GB,
        - extra latency budget available (~30ms).
        """
        # Visual fraction proxy from candidate set
        total = max(1, sum(len(lst) for lst in lists) if lists else len(visual_nodes))
        visual_frac = (len(visual_nodes) / total) if total else 0.0
        topk_ok = settings.retrieval.reranking_top_k <= COLPALI_TOPK_MAX
        vram_ok = _has_cuda_vram(COLPALI_MIN_VRAM_GB)
        # Allow ops override via env/setting (optional); default False
        ops_force = bool(getattr(settings.retrieval, "enable_colpali", False))
        return ops_force or (visual_frac >= 0.4 and topk_ok and vram_ok)

    # RRF merge handles fusion+dedupe; helper removed


__all__ = [
    "MultimodalReranker",
    "build_text_reranker",
    "build_visual_reranker",
]
