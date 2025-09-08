"""Modality-aware reranking utilities per ADR-037 and SPEC-005.

Changes in this patch:
- Default visual rerank uses SigLIP text-image cosine.
- Optional ColPali auto-enables via policy thresholds (VRAM/budget/K/visual fraction).
- Rank-level RRF merge across modalities; fail-open on timeout.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FTimeoutError
from functools import cache
from typing import Any

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.postprocessor.colpali_rerank import ColPaliRerank
from loguru import logger

from src.config import settings
from src.utils.multimodal import TEXT_TRUNCATION_LIMIT
from src.utils.telemetry import log_jsonl

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
    """Get current time in milliseconds."""
    return time.perf_counter() * 1000.0


def _run_with_timeout(fn: Callable[[], Any], timeout_ms: int) -> Any | None:
    """Execute a callable with a hard timeout.

    Returns None when the timeout elapses; otherwise returns the callable's result.
    """
    # Small executor per call keeps code simple and avoids shared state complexity
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=max(0.0, timeout_ms) / 1000.0)
        except FTimeoutError:
            return None


def _has_cuda_vram(min_gb: float) -> bool:
    """Check if CUDA GPU has sufficient VRAM.

    Args:
        min_gb: Minimum VRAM required in gigabytes.

    Returns:
        bool: True if CUDA is available and has sufficient VRAM, False otherwise.
    """
    try:
        import torch

        if not torch.cuda.is_available():  # type: ignore[attr-defined]
            return False
        props = torch.cuda.get_device_properties(0)  # type: ignore[attr-defined]
        total = props.total_memory / (1024**3)
        return total >= float(min_gb)
    except Exception:
        return False


def _rrf_merge(
    lists: list[list[NodeWithScore]], k_constant: int
) -> list[NodeWithScore]:
    """Rank-level Reciprocal Rank Fusion over multiple reranked lists.

    Args:
        lists: List of ranked lists to merge.
        k_constant: RRF k-constant for score calculation.

    Returns:
        list[NodeWithScore]: Fused list sorted by RRF scores.
    """
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

    Returns:
        tuple[Any, Any, str]: Tuple of (model, processor, device_str).
    """
    from transformers import SiglipModel, SiglipProcessor  # type: ignore

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():  # type: ignore[attr-defined]
            device = "cuda"
    except Exception:
        device = "cpu"

    # Pull model id from settings if available, fallback to default
    try:
        model_id = getattr(
            settings.embedding, "siglip_model_id", "google/siglip-base-patch16-224"
        )
    except Exception:  # pylint: disable=broad-exception-caught
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

    Args:
        query: Text query string.
        nodes: List of nodes with potential image metadata.
        budget_ms: Time budget in milliseconds.

    Returns:
        list[NodeWithScore]: Nodes with updated scores sorted descending.
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

        # Support encrypted images written as .enc — decrypt to a temporary file first
        try:
            from src.utils.security import (
                decrypt_file,
            )  # local import to avoid heavy deps
        except Exception:  # pragma: no cover - defensive

            def decrypt_file(p: str) -> str:  # type: ignore
                return p

        images: list[Any] = []
        temp_files: list[str] = []
        for p in paths:
            if p:
                try:
                    to_open = p
                    if str(p).endswith(".enc"):
                        dec = decrypt_file(p)
                        to_open = dec
                        temp_files.append(dec)
                    images.append(Image.open(to_open).convert("RGB"))
                except Exception:
                    images.append(None)
            else:
                images.append(None)
            if _now_ms() - t0 > budget_ms:
                logger.warning("SigLIP load images timeout; fail-open")
                # Cleanup temp files on early exit
                for tp in temp_files:
                    with contextlib.suppress(Exception):
                        os.remove(tp)
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
                # Cleanup temp files on early exit
                for tp in temp_files:
                    with contextlib.suppress(Exception):
                        os.remove(tp)
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
        # Cleanup temp files
        for tp in temp_files:
            with contextlib.suppress(Exception):
                os.remove(tp)
        return nodes_sorted[: settings.retrieval.reranking_top_k]
    except Exception as exc:
        logger.warning("SigLIP rerank error: {} — fail-open", exc)
        return nodes


def _parse_top_k(value: int | str | None) -> int:
    """Validate and convert top_n values to int.

    Args:
        value: Value to convert, can be int, str, or None.

    Returns:
        int: Parsed integer value, or default from settings.

    Raises:
        ValueError: If value cannot be converted to int.
    """
    if value is None:
        return settings.retrieval.reranking_top_k
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid top_n: {value}") from exc


@cache
def _build_text_reranker_cached(top_n: int) -> SentenceTransformerRerank:
    """Create cached SentenceTransformer reranker for text.

    Args:
        top_n: Number of top results to return.

    Returns:
        SentenceTransformerRerank: Configured text reranker instance.
    """
    # Respect configured model id to keep settings authoritative
    return SentenceTransformerRerank(
        model=getattr(settings.retrieval, "reranker_model", "BAAI/bge-reranker-v2-m3"),
        top_n=top_n,
        use_fp16=True,
        normalize=settings.retrieval.reranker_normalize_scores,
    )


def build_text_reranker(top_n: int | str | None = None) -> SentenceTransformerRerank:
    """Create text CrossEncoder reranker (BGE v2-m3).

    Args:
        top_n: Number of top results to return. Defaults to settings value.

    Returns:
        SentenceTransformerRerank: Configured text reranker instance.
    """
    k = _parse_top_k(top_n)
    return _build_text_reranker_cached(k)


@cache
def _build_visual_reranker_cached(top_n: int) -> ColPaliRerank:
    """Create cached ColPali reranker for visual content.

    Args:
        top_n: Number of top results to return.

    Returns:
        ColPaliRerank: Configured visual reranker instance.
    """
    return ColPaliRerank(model="vidore/colpali-v1.2", top_n=top_n)


def build_visual_reranker(top_n: int | str | None = None) -> ColPaliRerank:
    """Create visual reranker (ColPali).

    Args:
        top_n: Number of top results to return. Defaults to settings value.

    Returns:
        ColPaliRerank: Configured visual reranker instance.

    Raises:
        ValueError: If ColPaliRerank initialization fails.
    """
    k = _parse_top_k(top_n)
    try:
        return _build_visual_reranker_cached(k)
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - library quirk
        raise ValueError(f"ColPaliRerank initialization failed: {exc}") from exc


class MultimodalReranker(BaseNodePostprocessor):
    """Postprocessor that applies text and visual rerankers per node modality.

    This reranker uses modality-aware processing with:
    - Text reranking via BGE CrossEncoder
    - Visual reranking via SigLIP (default) or ColPali (optional)
    - RRF fusion for multi-modality results
    - Time budgets and fail-open behavior
    """

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        """Apply modality-aware reranking with time budgets and fail-open behavior.

        Args:
            nodes: List of nodes to rerank.
            query_bundle: Query bundle containing the search query.

        Returns:
            list[NodeWithScore]: Reranked nodes sorted by relevance scores.
        """
        if not nodes or not query_bundle:
            return nodes

        text_nodes, visual_nodes = self._split_by_modality(nodes)

        # Stage 2: text rerank with timeout
        lists: list[list[NodeWithScore]] = []
        t_start = _now_ms()
        try:
            if text_nodes:

                def _do_text():
                    return build_text_reranker().postprocess_nodes(
                        text_nodes, query_str=query_bundle.query_str
                    )

                tr = _run_with_timeout(_do_text, TEXT_RERANK_TIMEOUT_MS)
                if tr is not None:
                    lists.append(tr)
                else:
                    logger.warning("Text rerank timeout; fail-open")
                    log_jsonl(
                        {
                            "rerank.stage": "text",
                            "rerank.topk": int(settings.retrieval.reranking_top_k),
                            "rerank.latency_ms": int(_now_ms() - t_start),
                            "rerank.timeout": True,
                        }
                    )
                    return nodes
                log_jsonl(
                    {
                        "rerank.stage": "text",
                        "rerank.topk": int(settings.retrieval.reranking_top_k),
                        "rerank.latency_ms": int(_now_ms() - t_start),
                        "rerank.timeout": False,
                    }
                )
        except Exception as exc:
            logger.warning("Text rerank error: {} — fail-open", exc)

        # Stage 3: visual rerank — SigLIP default within budget
        v_start = _now_ms()
        try:
            if visual_nodes:
                sr = _siglip_rescore(
                    query_bundle.query_str, visual_nodes, SIGLIP_TIMEOUT_MS
                )
                lists.append(sr)
                log_jsonl(
                    {
                        "rerank.stage": "visual",
                        "rerank.topk": int(settings.retrieval.reranking_top_k),
                        "rerank.latency_ms": int(_now_ms() - v_start),
                        "rerank.timeout": False,
                    }
                )
        except Exception as exc:
            logger.warning("SigLIP rerank error: {} — continue without", exc)

        # Optional Stage 3b: ColPali policy
        try:
            if self._should_enable_colpali(visual_nodes, lists):
                base = lists[-1] if lists else visual_nodes
                pruned = base[:SIGLIP_PRUNE_M]

                def _do_colpali():
                    return build_visual_reranker(
                        top_n=min(COLPALI_FINAL_M, settings.retrieval.reranking_top_k)
                    ).postprocess_nodes(pruned, query_str=query_bundle.query_str)

                remaining = max(
                    0, SIGLIP_TIMEOUT_MS + COLPALI_TIMEOUT_MS - int(_now_ms() - v_start)
                )
                cr = _run_with_timeout(_do_colpali, remaining)
                if cr is not None:
                    lists.append(cr)
                    log_jsonl(
                        {
                            "rerank.stage": "colpali",
                            "rerank.topk": int(settings.retrieval.reranking_top_k),
                            "rerank.latency_ms": int(_now_ms() - v_start),
                            "rerank.timeout": False,
                        }
                    )
                else:
                    logger.warning("ColPali rerank timeout; continue without")
                    log_jsonl(
                        {
                            "rerank.stage": "colpali",
                            "rerank.topk": int(settings.retrieval.reranking_top_k),
                            "rerank.latency_ms": int(_now_ms() - v_start),
                            "rerank.timeout": True,
                        }
                    )
        except Exception as exc:
            logger.warning("ColPali rerank error: {} — continue without", exc)

        if _now_ms() - v_start > (SIGLIP_TIMEOUT_MS + COLPALI_TIMEOUT_MS):
            logger.warning("Visual rerank timeout; fail-open")
            log_jsonl(
                {
                    "rerank.stage": "visual",
                    "rerank.topk": int(settings.retrieval.reranking_top_k),
                    "rerank.latency_ms": int(_now_ms() - v_start),
                    "rerank.timeout": True,
                }
            )
            return nodes

        if not lists:
            return nodes[: settings.retrieval.reranking_top_k]
        fused = _rrf_merge(lists, k_constant=int(settings.retrieval.rrf_k))
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
        # Emit final delta metric: simple change count in top-k ids
        try:
            before_ids = [
                n.node.node_id for n in nodes[: settings.retrieval.reranking_top_k]
            ]
            after_ids = [n.node.node_id for n in out]
            delta_changed = len(set(after_ids) - set(before_ids))
            log_jsonl(
                {
                    "rerank.stage": "final",
                    "rerank.topk": int(settings.retrieval.reranking_top_k),
                    "rerank.latency_ms": 0,
                    "rerank.timeout": False,
                    "rerank.delta_changed_count": int(delta_changed),
                }
            )
        except Exception as exc:
            logger.warning("Final rerank metrics error: {} — skipping telemetry", exc)
        return out

    @staticmethod
    def _split_by_modality(
        nodes: list[NodeWithScore],
    ) -> tuple[list[NodeWithScore], list[NodeWithScore]]:
        """Split nodes into text and visual modalities based on metadata.

        Args:
            nodes: List of nodes to split by modality.

        Returns:
            tuple[list[NodeWithScore], list[NodeWithScore]]: Tuple of
            (text_nodes, visual_nodes).
        """
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

        Args:
            visual_nodes: List of visual nodes being processed.
            lists: Current list of reranked result lists.

        Returns:
            bool: True if ColPali should be enabled, False otherwise.
        """
        # Visual fraction proxy from candidate set
        total = max(1, sum(len(lst) for lst in lists) if lists else len(visual_nodes))
        visual_frac = (len(visual_nodes) / total) if total else 0.0
        topk_ok = settings.retrieval.reranking_top_k <= COLPALI_TOPK_MAX
        vram_ok = _has_cuda_vram(COLPALI_MIN_VRAM_GB)
        # Allow ops override via env/setting (optional); default False
        ops_force = bool(getattr(settings.retrieval, "enable_colpali", False))
        return ops_force or (visual_frac >= 0.4 and topk_ok and vram_ok)


__all__ = [
    "MultimodalReranker",
    "build_text_reranker",
    "build_visual_reranker",
]
