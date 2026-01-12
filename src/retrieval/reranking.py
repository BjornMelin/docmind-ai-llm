"""Modality-aware reranking utilities per ADR-037 and SPEC-005.

Changes in this patch:
- Default visual rerank uses SigLIP text-image cosine.
- Optional ColPali auto-enables via policy thresholds (VRAM/budget/K/visual fraction).
- Rank-level RRF merge across modalities; fail-open on timeout.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FTimeoutError
from functools import cache
from typing import Any

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

# ColPali import is optional; import inside builder to avoid hard dependency
from loguru import logger

from src.config import settings
from src.persistence.artifacts import ArtifactRef, ArtifactStore
from src.retrieval.rrf import rrf_merge
from src.utils.core import has_cuda_vram, resolve_device
from src.utils.telemetry import log_jsonl


# Back-compat: tests and callers still import _rrf_merge from this module.
def _rrf_merge(lists: list[list[Any]], k_constant: int) -> list[Any]:
    """Compatibility wrapper for tests and legacy callers.

    Accepts NodeWithScore lists or lightweight stand-ins used in tests.
    """
    scores: dict[str, tuple[float, Any]] = {}
    for ranked in lists:
        for rank, item in enumerate(ranked, start=1):
            node = getattr(item, "node", None)
            nid = (
                getattr(node, "node_id", None)
                or getattr(node, "id_", None)
                or str(node)
            )
            inc = 1.0 / (k_constant + rank)
            cur = scores.get(str(nid))
            if cur is None:
                scores[str(nid)] = (inc, item)
            else:
                scores[str(nid)] = (cur[0] + inc, cur[1])

    fused_list = list(scores.values())
    fused_list.sort(
        key=lambda t: (-float(t[0]), str(getattr(t[1].node, "node_id", "")))
    )

    out: list[Any] = []
    for score, item in fused_list:
        if isinstance(item, NodeWithScore):
            out.append(NodeWithScore(node=item.node, score=float(score)))
        else:
            with contextlib.suppress(Exception):
                item.score = float(score)
            out.append(item)
    return out


# Time budgets (ms)
def _text_timeout_ms() -> int:
    try:
        return int(settings.retrieval.text_rerank_timeout_ms)
    except (
        AttributeError,
        TypeError,
        ValueError,
    ):  # pragma: no cover - defensive default
        return 250


def _siglip_timeout_ms() -> int:
    try:
        return int(settings.retrieval.siglip_timeout_ms)
    except (
        AttributeError,
        TypeError,
        ValueError,
    ):  # pragma: no cover - defensive default
        return 150


def _colpali_timeout_ms() -> int:
    try:
        return int(settings.retrieval.colpali_timeout_ms)
    except (
        AttributeError,
        TypeError,
        ValueError,
    ):  # pragma: no cover - defensive default
        return 400


# Policy thresholds
COLPALI_MIN_VRAM_GB = 8.0  # enable when >= 8-12 GB
COLPALI_TOPK_MAX = 16  # small-K scenarios
SIGLIP_PRUNE_M = 64  # cascade: SigLIP prune to m → ColPali m'
COLPALI_FINAL_M = 16


def _now_ms() -> float:
    """Get current time in milliseconds."""
    return time.perf_counter() * 1000.0


def _run_with_timeout(fn: Callable[[], Any], timeout_ms: int) -> Any | None:
    """Execute a callable with a hard timeout (best effort, non-blocking).

    Returns None when the timeout elapses; otherwise returns the callable's
    result. Ensures the worker future is cancelled and the executor is shut
    down without waiting so the caller does not block on long-running tasks.
    """
    # Choose executor based on settings (default: thread). Process executors
    # require picklable callables; fall back to thread if submission fails.
    executor = None
    executors: list[ThreadPoolExecutor | ProcessPoolExecutor] = []
    try:
        exec_type = getattr(
            getattr(settings, "retrieval", object()), "rerank_executor", "thread"
        )
        if exec_type == "process":
            try:
                executor = ProcessPoolExecutor(max_workers=1)
                executors.append(executor)
                fut = executor.submit(fn)
            except (
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:  # pragma: no cover - pickling/env edge cases
                logger.warning(
                    "Process executor unsupported; falling back to thread: {}",
                    exc,
                )
                executor = ThreadPoolExecutor(max_workers=1)
                executors.append(executor)
                fut = executor.submit(fn)
        else:
            executor = ThreadPoolExecutor(max_workers=1)
            executors.append(executor)
            fut = executor.submit(fn)
    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:  # defensive: ensure we have an executor
        logger.warning(
            "Executor initialization failed: {} — using thread fallback", exc
        )
        executor = ThreadPoolExecutor(max_workers=1)
        executors.append(executor)
        fut = executor.submit(fn)
    try:
        return fut.result(timeout=max(0.0, timeout_ms) / 1000.0)
    except FTimeoutError:
        with contextlib.suppress(RuntimeError):
            fut.cancel()
        return None
    finally:
        for ex in executors:
            with contextlib.suppress(RuntimeError, OSError):
                ex.shutdown(wait=False, cancel_futures=True)


# removed local _has_cuda_vram wrapper; use core.has_cuda_vram directly


def _compute_siglip_scores(
    model: Any,
    processor: Any,
    device: str,
    tfeat: Any,
    images: list[Any],
    *,
    budget_ms: int,
    start_ms: float,
    batch_size: int,
) -> list[float] | None:
    """Batched SigLIP cosine scores with timeout guard."""
    try:
        import torch  # type: ignore
    except (
        ImportError,
        ModuleNotFoundError,
    ):  # pragma: no cover - torch optional in CI
        return [float("-inf")] * len(images)

    feats: list[float] = [float("-inf")] * len(images)
    valid: list[tuple[int, Any]] = [
        (i, img) for i, img in enumerate(images) if img is not None
    ]
    bs = max(1, int(batch_size))
    for j in range(0, len(valid), bs):
        if _now_ms() - start_ms > budget_ms:
            return None
        idxs, batch_imgs = zip(*valid[j : j + bs], strict=False)
        im_inputs = processor(images=list(batch_imgs), return_tensors="pt")
        if device == "cuda":
            im_inputs["pixel_values"] = im_inputs["pixel_values"].to("cuda")
        elif device == "mps":
            im_inputs["pixel_values"] = im_inputs["pixel_values"].to("mps")
        with torch.no_grad():  # type: ignore[name-defined]
            if hasattr(model, "get_image_features"):
                imfeat = model.get_image_features(**im_inputs)
                imfeat = imfeat / imfeat.norm(dim=-1, keepdim=True)
                sims = (tfeat @ imfeat.T).squeeze(0).detach().cpu().numpy().tolist()
            else:
                sims = [float("-inf")] * len(idxs)
        for k, s in zip(idxs, sims, strict=False):
            feats[int(k)] = float(s)
    return feats


def _load_siglip() -> tuple[Any, Any, str]:  # (model, processor, device)
    """Load SigLIP via shared utility to keep behavior consistent."""
    try:
        model_id = getattr(
            settings.embedding, "siglip_model_id", "google/siglip-base-patch16-224"
        )
    except AttributeError:
        model_id = "google/siglip-base-patch16-224"
    from src.utils.vision_siglip import load_siglip

    return load_siglip(model_id=model_id, device=None)


def _extract_image_paths(ns: list[NodeWithScore]) -> list[str]:
    """Extract possible image paths from node metadata and attributes."""
    out: list[str] = []

    store: ArtifactStore | None = None
    for nn in ns:
        meta = getattr(nn.node, "metadata", {}) or {}
        p = ""
        # Final-release: resolve local paths from stable artifact refs rather than
        # accepting raw filesystem paths from node metadata.
        img_id = meta.get("image_artifact_id")
        img_sfx = meta.get("image_artifact_suffix")
        if not img_id:
            img_id = meta.get("thumbnail_artifact_id")
            img_sfx = meta.get("thumbnail_artifact_suffix")
        if img_id:
            try:
                if store is None:
                    store = ArtifactStore.from_settings(settings)
                ref = ArtifactRef(sha256=str(img_id), suffix=str(img_sfx or ""))
                p = str(store.resolve_path(ref))
            except Exception:
                p = ""
        out.append(str(p) if p else "")
    return out


def _load_images_for_siglip(
    paths: list[str], start_ms: float, budget: int
) -> list[Any]:
    """Load images with timeout guard using encrypted-aware opener.

    Returns a list of PIL images (RGB) or None placeholders for missing paths.
    May early-return if the timeout budget is exceeded; caller decides policy.
    """
    imgs: list[Any] = []
    from src.utils.images import open_image_encrypted

    for pth in paths:
        if pth:
            try:
                with open_image_encrypted(pth) as _im:
                    imgs.append(_im.convert("RGB").copy() if _im is not None else None)
            except (OSError, ValueError, RuntimeError, TypeError, ImportError):
                imgs.append(None)
        else:
            imgs.append(None)
        if _now_ms() - start_ms > budget:
            return imgs
    return imgs


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
    images: list[Any] = []
    try:
        import torch  # local import to avoid global dependency

        # Gather and load images
        paths = _extract_image_paths(nodes)
        images = _load_images_for_siglip(paths, t0, budget_ms)
        if _now_ms() - t0 > budget_ms:
            logger.warning("SigLIP load images timeout; fail-open")
            return nodes
        model, processor, device = _load_siglip()
        # Text features (1, D)
        txt_inputs = processor(text=[query], padding="max_length", return_tensors="pt")
        if device == "cuda":
            for k, v in txt_inputs.items():
                txt_inputs[k] = v.to("cuda")
        elif device == "mps":
            for k, v in txt_inputs.items():
                txt_inputs[k] = v.to("mps")
        with torch.no_grad():  # type: ignore[name-defined]
            tfeat = model.get_text_features(**txt_inputs)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        # Batched image features via helper
        try:
            bs_conf = int(getattr(settings.retrieval, "siglip_batch_size", 0))
        except (AttributeError, ValueError, TypeError):
            bs_conf = 0
        bs = bs_conf if bs_conf > 0 else (8 if device == "cuda" else 2)
        feats_or_none = _compute_siglip_scores(
            model,
            processor,
            device,
            tfeat,
            images,
            budget_ms=budget_ms,
            start_ms=t0,
            batch_size=bs,
        )
        if feats_or_none is None:
            logger.warning("SigLIP compute timeout; fail-open")
            return nodes
        feats = feats_or_none
        # Update scores and sort
        for n, s in zip(nodes, feats, strict=False):
            n.score = s
        # Deterministic sorting: score desc, id asc
        nodes_sorted = sorted(
            nodes,
            key=lambda x: (-float(x.score or 0.0), str(getattr(x.node, "node_id", ""))),
        )
        # Pre-fusion prune M for visual stage
        try:
            prune_m = int(getattr(settings.retrieval, "siglip_prune_m", SIGLIP_PRUNE_M))
        except (AttributeError, ValueError, TypeError):
            prune_m = SIGLIP_PRUNE_M
        return nodes_sorted[: max(1, prune_m)]
    except (RuntimeError, ValueError, OSError, TypeError) as exc:
        logger.warning("SigLIP rerank error: {} — fail-open", exc)
        return nodes
    finally:
        # Always cleanup images
        for img in images:
            with contextlib.suppress(OSError, AttributeError, RuntimeError):
                if hasattr(img, "close"):
                    img.close()


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
    """Create cached SentenceTransformer reranker for text."""
    return SentenceTransformerRerank(
        model=getattr(settings.retrieval, "reranker_model", "BAAI/bge-reranker-v2-m3"),
        top_n=top_n,
    )


def build_text_reranker(top_n: int | str | None = None) -> SentenceTransformerRerank:
    """Create text CrossEncoder reranker (BGE v2-m3) via LlamaIndex.

    Returns:
        SentenceTransformerRerank implementing postprocess_nodes.
    """
    k = _parse_top_k(top_n)
    try:
        return _build_text_reranker_cached(k)
    except OSError as exc:  # offline HF hub in CI or local
        logger.warning("Text reranker offline; using NoOpTextReranker: {}", exc)
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
        logger.warning("Text reranker init failed; using NoOpTextReranker: {}", exc)

    class NoOpTextReranker:  # minimal LlamaIndex-like interface
        """Fallback reranker that returns the first ``top_n`` nodes."""

        def __init__(self, top_n: int) -> None:
            """Store the number of nodes to retain."""
            self.top_n = int(top_n)

        def postprocess_nodes(
            self, nodes: list[NodeWithScore], **_: Any
        ) -> list[NodeWithScore]:
            """Return the leading ``top_n`` nodes unchanged."""
            return nodes[: self.top_n]

    return NoOpTextReranker(k)  # type: ignore[return-value]


@cache
def _build_visual_reranker_cached(top_n: int) -> Any:
    """Create cached ColPali reranker for visual content.

    Args:
        top_n: Number of top results to return.

    Returns:
        ColPaliRerank: Configured visual reranker instance.
    """
    try:
        import importlib

        mod = importlib.import_module("llama_index.postprocessor.colpali_rerank")
        colpali_rerank_cls = mod.ColPaliRerank
    except (ImportError, AttributeError, RuntimeError) as exc:
        raise ValueError("ColPaliRerank not available") from exc
    return colpali_rerank_cls(model="vidore/colpali-v1.2", top_n=top_n)


def build_visual_reranker(top_n: int | str | None = None) -> Any:
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
        # Stage 2: text rerank with timeout (outer guard) + cooperative batches
        lists: list[list[NodeWithScore]] = []

        def _run_stage(
            name: str, fn: Callable[[], list[NodeWithScore]], timeout_ms: int
        ) -> list[NodeWithScore] | None:
            s = _now_ms()
            res = _run_with_timeout(fn, timeout_ms)
            log_jsonl(
                {
                    "rerank.stage": name,
                    "rerank.topk": int(settings.retrieval.reranking_top_k),
                    "rerank.latency_ms": int(_now_ms() - s),
                    "rerank.timeout": res is None,
                }
            )
            return res

        try:
            if text_nodes:
                reranker = build_text_reranker(top_n=settings.retrieval.reranking_top_k)

                def _do_text():
                    return reranker.postprocess_nodes(
                        text_nodes, query_str=query_bundle.query_str
                    )

                tr = _run_stage("text", _do_text, _text_timeout_ms() + 50)
                if tr is None:
                    logger.warning("Text rerank timeout; fail-open")
                    return nodes
                lists.append(tr)
        except (RuntimeError, ValueError, OSError, TypeError) as exc:
            logger.warning("Text rerank error: {} — fail-open", exc)
        # Stage 3: visual rerank — SigLIP default within budget
        v_start = _now_ms()
        # Visual stage runs whenever visual nodes exist (no UI toggle)
        if visual_nodes:
            try:
                sr = _siglip_rescore(
                    query_bundle.query_str, visual_nodes, _siglip_timeout_ms()
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
            except (RuntimeError, ValueError, OSError, TypeError, ImportError) as exc:
                logger.warning("SigLIP rerank error: {} — continue without", exc)
            # Optional Stage 3b: ColPali policy (only when visual nodes exist)
            try:
                if self._should_enable_colpali(visual_nodes, lists):
                    base = lists[-1] if lists else visual_nodes
                    try:
                        prune_m = int(
                            getattr(
                                settings.retrieval, "siglip_prune_m", SIGLIP_PRUNE_M
                            )
                        )
                    except (AttributeError, ValueError, TypeError):
                        prune_m = SIGLIP_PRUNE_M
                    pruned = base[: max(1, prune_m)]

                    def _do_colpali():
                        return build_visual_reranker(
                            top_n=min(
                                COLPALI_FINAL_M, settings.retrieval.reranking_top_k
                            )
                        ).postprocess_nodes(pruned, query_str=query_bundle.query_str)

                    remaining = max(
                        0,
                        _siglip_timeout_ms()
                        + _colpali_timeout_ms()
                        - int(_now_ms() - v_start),
                    )
                    cr = _run_stage("colpali", _do_colpali, remaining)
                    if cr is None:
                        logger.warning("ColPali rerank timeout; continue without")
                    else:
                        lists.append(cr)
            except (RuntimeError, ValueError) as exc:
                logger.warning("ColPali rerank error: {} — continue without", exc)
            # Only enforce visual-stage timeout when visual processing attempted
            if _now_ms() - v_start > (_siglip_timeout_ms() + _colpali_timeout_ms()):
                logger.warning("Visual rerank timeout; fail-open")
                log_jsonl(
                    {
                        "rerank.stage": "visual",
                        "rerank.topk": int(settings.retrieval.reranking_top_k),
                        "rerank.latency_ms": int(_now_ms() - v_start),
                        "rerank.timeout": True,
                        "rerank.executor": getattr(
                            settings.retrieval, "rerank_executor", "thread"
                        ),
                    }
                )
                return nodes
        if not lists:
            return nodes[: settings.retrieval.reranking_top_k]
        fused = rrf_merge(lists, k_constant=int(settings.retrieval.rrf_k))
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
            configured_k = settings.retrieval.reranking_top_k
            effective_k = min(10, configured_k)
            before_ids = [n.node.node_id for n in nodes[:effective_k]]
            after_ids = [n.node.node_id for n in out[:effective_k]]
            before_positions = {nid: idx for idx, nid in enumerate(before_ids)}
            mrr_before = 0.0
            for idx in range(min(effective_k, len(before_ids))):
                mrr_before += 1.0 / float(idx + 1)
            mrr_after = 0.0
            for _rank, node_id in enumerate(after_ids, start=1):
                if node_id in before_positions:
                    mrr_after += 1.0 / float(before_positions[node_id] + 1)
            delta_mrr = mrr_after - mrr_before
            path = (
                "both"
                if (bool(text_nodes) and bool(visual_nodes))
                else (
                    "text"
                    if bool(text_nodes)
                    else ("visual" if bool(visual_nodes) else "none")
                )
            )
            # simple score stats
            try:
                scores = [float(n.score or 0.0) for n in out]
            except (TypeError, ValueError):  # pragma: no cover - defensive
                scores = []
            score_mean = float(sum(scores) / len(scores)) if scores else 0.0
            score_max = float(max(scores)) if scores else 0.0
            log_jsonl(
                {
                    "rerank.stage": "final",
                    "rerank.topk": int(configured_k),
                    "rerank.latency_ms": 0,
                    "rerank.timeout": False,
                    "rerank.delta_mrr_at_10": delta_mrr,
                    "rerank.path": path,
                    "rerank.total_timeout_budget_ms": int(
                        getattr(settings.retrieval, "total_rerank_budget_ms", 0)
                        or (
                            _text_timeout_ms()
                            + _siglip_timeout_ms()
                            + _colpali_timeout_ms()
                        )
                    ),
                    "rerank.executor": getattr(
                        settings.retrieval, "rerank_executor", "thread"
                    ),
                    "rerank.input_count": len(nodes),
                    "rerank.output_count": len(out),
                    "rerank.score.mean": score_mean,
                    "rerank.score.max": score_max,
                }
            )
        except (RuntimeError, ValueError, OSError, TypeError) as exc:
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

        def _meta_of(nod: NodeWithScore) -> dict:
            return getattr(nod.node, "metadata", {}) or {}

        text = [n for n in nodes if _meta_of(n).get("modality", "text") == "text"]
        visual = [
            n
            for n in nodes
            if _meta_of(n).get("modality") in {"image", "pdf_page_image"}
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
        dev_str, dev_idx = resolve_device("auto")
        _ = dev_str  # reserved for future routing
        vram_ok = has_cuda_vram(COLPALI_MIN_VRAM_GB, device_index=int(dev_idx or 0))
        # Allow ops override via env/setting (optional); default False
        ops_force = bool(getattr(settings.retrieval, "enable_colpali", False))
        return ops_force or (visual_frac >= 0.4 and topk_ok and vram_ok)


__all__ = [
    "MultimodalReranker",
    "build_text_reranker",
    "build_visual_reranker",
]


# ---- Helpers for factories (DRY) ----
def get_postprocessors(
    mode: str, *, use_reranking: bool, top_n: int | None = None
) -> list | None:
    """Return node_postprocessors list for the given mode or None.

    Args:
        mode: One of "vector", "hybrid", or "kg".
        use_reranking: Global toggle; when False returns None.
        top_n: Optional top_n for text reranker (KG).

    Returns:
        list | None: List of postprocessors or None when disabled/unavailable.
    """
    if not use_reranking:
        return None
    try:
        if mode in ("vector", "hybrid"):
            return [MultimodalReranker()]
        if mode == "kg":
            return [
                build_text_reranker(
                    top_n=top_n
                    if top_n is not None
                    else settings.retrieval.reranking_top_k
                )
            ]
    except (ValueError, TypeError):  # defensive: score cast
        return None
    return None
