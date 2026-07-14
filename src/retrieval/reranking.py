"""Modality-aware reranking utilities per ADR-037 and SPEC-005.

Changes in this patch:
- Default visual rerank uses SigLIP text-image cosine.
- Rank-level RRF merge across modalities; fail-open on timeout.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeoutError
from functools import cache
from pathlib import Path
from typing import Any, TypeVar

from huggingface_hub import snapshot_download
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from loguru import logger
from pydantic import Field, PrivateAttr

from src.config import settings
from src.config.embedding_defaults import (
    DEFAULT_BGE_RERANKER_MODEL_ID,
    DEFAULT_BGE_RERANKER_MODEL_REVISION,
)
from src.persistence.artifacts import ArtifactRef, ArtifactStore
from src.retrieval.async_work import (
    AsyncWorkCapacityError,
    AsyncWorkClosedError,
    AsyncWorkExecutor,
)
from src.retrieval.rrf import rrf_merge
from src.utils.log_safety import build_pii_log_entry
from src.utils.telemetry import log_jsonl
from src.utils.vision_siglip import siglip_features

_StageResult = TypeVar("_StageResult")


def _emit_rerank_telemetry(event: dict[str, Any]) -> None:
    """Record best-effort telemetry without changing reranking behavior."""
    try:
        log_jsonl(event)
    except (RuntimeError, ValueError, OSError, TypeError) as exc:
        redaction = build_pii_log_entry(str(exc), key_id="reranking.telemetry")
        logger.warning(
            "Rerank telemetry failed; continuing (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )


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


def _total_rerank_budget_ms() -> int:
    try:
        configured = int(settings.retrieval.total_rerank_budget_ms)
    except (AttributeError, TypeError, ValueError):
        configured = 0
    return configured or (_text_timeout_ms() + _siglip_timeout_ms())


def _remaining_rerank_budget_ms(deadline: float) -> float:
    """Return the remaining monotonic rerank budget in milliseconds."""
    return max(0.0, (deadline - time.monotonic()) * 1000.0)


def _capped_stage_budget_ms(deadline: float, stage_budget_ms: int) -> float:
    """Cap one stage timeout by the remaining total rerank budget."""
    return min(float(stage_budget_ms), _remaining_rerank_budget_ms(deadline))


def _isolate_rerank_nodes(
    nodes: list[NodeWithScore],
) -> list[NodeWithScore]:
    """Copy mutable score and metadata owners before native worker execution."""
    isolated: list[NodeWithScore] = []
    for node_with_score in nodes:
        node = node_with_score.node.model_copy(
            update={"metadata": dict(node_with_score.node.metadata)}
        )
        isolated.append(node_with_score.model_copy(update={"node": node}))
    return isolated


def _get_siglip_prune_m() -> int:
    """Return SigLIP prune threshold with defensive defaults."""
    try:
        return int(getattr(settings.retrieval, "siglip_prune_m", SIGLIP_PRUNE_M))
    except (AttributeError, ValueError, TypeError):
        return SIGLIP_PRUNE_M


SIGLIP_PRUNE_M = 64


def _now_ms() -> float:
    """Get current time in milliseconds."""
    return time.perf_counter() * 1000.0


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
                try:
                    imfeat = siglip_features(model.get_image_features(**im_inputs))
                    sims = (tfeat @ imfeat.T).squeeze(0).detach().cpu().numpy().tolist()
                except (
                    ImportError,
                    ModuleNotFoundError,
                    AttributeError,
                    RuntimeError,
                    ValueError,
                    OSError,
                    TypeError,
                ) as exc:
                    redaction = build_pii_log_entry(
                        str(exc), key_id="reranking.siglip.fail_open"
                    )
                    logger.warning(
                        "SigLIP image feature error; "
                        "fail-open (error_type={}, error={})",
                        type(exc).__name__,
                        redaction.redacted,
                    )
                    sims = [float("-inf")] * len(idxs)
            else:
                sims = [float("-inf")] * len(idxs)
        for k, s in zip(idxs, sims, strict=False):
            feats[int(k)] = float(s)
    return feats


def _load_siglip() -> tuple[Any, Any, str]:  # (model, processor, device)
    """Load SigLIP via shared utility to keep behavior consistent."""
    try:
        embedding_settings = settings.embedding
        model_id = getattr(
            embedding_settings, "siglip_model_id", "google/siglip-base-patch16-224"
        )
        revision = getattr(
            embedding_settings,
            "siglip_model_revision",
            None,
        )
        cache_folder = getattr(embedding_settings, "cache_folder", None)
    except AttributeError:
        model_id = "google/siglip-base-patch16-224"
        revision = None
        cache_folder = None
    from src.utils.vision_siglip import load_siglip

    return load_siglip(
        model_id=model_id,
        device=None,
        revision=revision,
        cache_folder=cache_folder,
    )


def _extract_image_paths(ns: list[NodeWithScore]) -> list[str]:
    """Extract possible image paths from node metadata and attributes."""
    out: list[str] = []

    store: ArtifactStore | None = None  # Lazy init on first artifact ref.
    for nn in ns:
        meta = getattr(nn.node, "metadata", {}) or {}
        p = ""
        # Resolve local paths from stable artifact refs rather than accepting raw
        # filesystem paths from node metadata.
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
            except (ValueError, FileNotFoundError, OSError, AttributeError):
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
                with open_image_encrypted(pth) as im:
                    imgs.append(im.convert("RGB").copy() if im is not None else None)
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
            try:
                tfeat = siglip_features(model.get_text_features(**txt_inputs))
            except (
                ImportError,
                ModuleNotFoundError,
                AttributeError,
                RuntimeError,
                ValueError,
                OSError,
                TypeError,
            ) as exc:
                redaction = build_pii_log_entry(
                    str(exc), key_id="reranking.siglip.fail_open"
                )
                logger.warning(
                    "SigLIP text feature error; fail-open (error_type={}, error={})",
                    type(exc).__name__,
                    redaction.redacted,
                )
                return nodes
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
        # Bound the visual list before rank fusion.
        prune_m = _get_siglip_prune_m()
        return nodes_sorted[: max(1, prune_m)]
    except (RuntimeError, ValueError, OSError, TypeError) as exc:
        redaction = build_pii_log_entry(str(exc), key_id="reranking.siglip.fail_open")
        logger.warning(
            "SigLIP rerank error; fail-open (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )
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
def _build_text_reranker_cached(
    top_n: int,
    model: str,
    cache_folder: str,
) -> SentenceTransformerRerank:
    """Create cached SentenceTransformer reranker for text."""
    model_source = model
    if model == DEFAULT_BGE_RERANKER_MODEL_ID:
        model_source = snapshot_download(
            repo_id=model,
            revision=DEFAULT_BGE_RERANKER_MODEL_REVISION,
            cache_dir=cache_folder,
            local_files_only=True,
        )
    return SentenceTransformerRerank(
        model=model_source,
        top_n=top_n,
        trust_remote_code=False,
    )


def build_text_reranker(top_n: int | str | None = None) -> SentenceTransformerRerank:
    """Create text CrossEncoder reranker (BGE v2-m3) via LlamaIndex.

    Returns:
        SentenceTransformerRerank implementing postprocess_nodes.
    """
    k = _parse_top_k(top_n)
    model = str(
        getattr(settings.retrieval, "reranker_model", DEFAULT_BGE_RERANKER_MODEL_ID)
    )
    embedding_settings = getattr(settings, "embedding", None)
    configured_cache = getattr(embedding_settings, "cache_folder", "./models_cache")
    cache_folder = str(Path(configured_cache).expanduser().resolve())
    try:
        return _build_text_reranker_cached(k, model, cache_folder)
    except OSError as exc:  # offline HF hub in CI or local
        redaction = build_pii_log_entry(str(exc), key_id="reranking.text.offline")
        logger.warning(
            "Text reranker offline; using NoOpTextReranker (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
        redaction = build_pii_log_entry(str(exc), key_id="reranking.text.init")
        logger.warning(
            "Text reranker init failed; using NoOpTextReranker "
            "(error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )

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


class MultimodalReranker(BaseNodePostprocessor):
    """Postprocessor that applies text and visual rerankers per node modality.

    This reranker uses modality-aware processing with:
    - Text reranking via BGE CrossEncoder
    - Visual reranking via SigLIP
    - RRF fusion for multi-modality results
    - Time budgets and fail-open behavior
    """

    top_n: int = Field(
        default_factory=lambda: int(settings.retrieval.reranking_top_k),
        ge=1,
    )
    _work: AsyncWorkExecutor = PrivateAttr(
        default_factory=lambda: AsyncWorkExecutor(name="docmind-rerank-cpu")
    )

    def close(self) -> None:
        """Reject new reranking work without waiting for an active stage."""
        self._work.close()

    async def aclose(self) -> None:
        """Reject new work and drain an active reranking stage."""
        await self._work.aclose()

    def _run_stage(
        self,
        name: str,
        fn: Callable[[], _StageResult],
        timeout_ms: float,
    ) -> _StageResult | None:
        """Run a rerank stage with timeout telemetry."""
        started = _now_ms()
        try:
            result = self._work.run_sync(
                fn,
                timeout=max(0.0, timeout_ms) / 1000.0,
            )
        except (
            FutureTimeoutError,
            AsyncWorkCapacityError,
            AsyncWorkClosedError,
        ):
            result = None
        _emit_rerank_telemetry(
            {
                "rerank.stage": name,
                "rerank.topk": self.top_n,
                "rerank.latency_ms": int(_now_ms() - started),
                "rerank.timeout": result is None,
            }
        )
        return result

    async def _arun_stage(
        self,
        name: str,
        fn: Callable[[], _StageResult],
        timeout_ms: int,
    ) -> _StageResult | None:
        """Run one async stage without using asyncio's default executor."""
        started = _now_ms()
        try:
            async with asyncio.timeout(max(0.0, timeout_ms) / 1000.0):
                result = await self._work.run(fn)
        except (TimeoutError, AsyncWorkCapacityError, AsyncWorkClosedError):
            result = None
        _emit_rerank_telemetry(
            {
                "rerank.stage": name,
                "rerank.topk": self.top_n,
                "rerank.latency_ms": int(_now_ms() - started),
                "rerank.timeout": result is None,
            }
        )
        return result

    def _text_result(
        self, text_nodes: list[NodeWithScore], query_bundle: QueryBundle
    ) -> list[NodeWithScore]:
        """Compute text reranking without owning execution policy."""
        reranker = build_text_reranker(top_n=self.top_n)
        return reranker.postprocess_nodes(
            text_nodes,
            query_str=query_bundle.query_str,
        )

    def _run_text_stage(
        self,
        text_nodes: list[NodeWithScore],
        query_bundle: QueryBundle,
        *,
        timeout_ms: float,
    ) -> list[NodeWithScore] | None:
        """Execute text reranking stage."""
        if not text_nodes:
            return []
        return self._run_stage(
            "text",
            lambda: self._text_result(text_nodes, query_bundle),
            timeout_ms,
        )

    async def _arun_text_stage(
        self, text_nodes: list[NodeWithScore], query_bundle: QueryBundle
    ) -> list[NodeWithScore] | None:
        """Execute text reranking on the owned async worker."""
        if not text_nodes:
            return []
        return await self._arun_stage(
            "text",
            lambda: self._text_result(text_nodes, query_bundle),
            _text_timeout_ms() + 50,
        )

    def _visual_result(
        self,
        visual_nodes: list[NodeWithScore],
        query_bundle: QueryBundle,
        lists: list[list[NodeWithScore]],
    ) -> list[list[NodeWithScore]]:
        """Compute visual reranking without owning execution policy."""
        results = list(lists)
        try:
            sr = _siglip_rescore(
                query_bundle.query_str, visual_nodes, _siglip_timeout_ms()
            )
            results.append(sr)
        except (RuntimeError, ValueError, OSError, TypeError, ImportError) as exc:
            redaction = build_pii_log_entry(
                str(exc), key_id="reranking.siglip.continue"
            )
            logger.warning(
                "SigLIP rerank error; continuing without (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )

        return results

    def _run_visual_stage(
        self,
        visual_nodes: list[NodeWithScore],
        query_bundle: QueryBundle,
        lists: list[list[NodeWithScore]],
        *,
        timeout_ms: float,
    ) -> tuple[list[list[NodeWithScore]], bool]:
        """Execute visual reranking stages and return timeout status."""
        if not visual_nodes:
            return lists, False
        try:
            result = self._run_stage(
                "visual",
                lambda: self._visual_result(visual_nodes, query_bundle, lists),
                timeout_ms,
            )
        except Exception as exc:
            redaction = build_pii_log_entry(
                str(exc), key_id="reranking.visual.fail_open"
            )
            logger.warning(
                "Visual rerank error; fail-open (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            return lists, True
        if result is None:
            logger.warning("Visual rerank timeout; fail-open")
            return lists, True
        return result, False

    async def _arun_visual_stage(
        self,
        visual_nodes: list[NodeWithScore],
        query_bundle: QueryBundle,
        lists: list[list[NodeWithScore]],
    ) -> tuple[list[list[NodeWithScore]], bool]:
        """Execute visual reranking on the owned async worker."""
        if not visual_nodes:
            return lists, False
        try:
            result = await self._arun_stage(
                "visual",
                lambda: self._visual_result(visual_nodes, query_bundle, lists),
                _siglip_timeout_ms(),
            )
        except Exception as exc:
            redaction = build_pii_log_entry(
                str(exc), key_id="reranking.visual.fail_open"
            )
            logger.warning(
                "Visual rerank error; fail-open (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            return lists, True
        if result is None:
            logger.warning("Visual rerank timeout; fail-open")
            return lists, True
        return result, False

    def _fuse_and_dedup(self, lists: list[list[NodeWithScore]]) -> list[NodeWithScore]:
        """Fuse rerank results and deduplicate nodes."""
        fused = rrf_merge(lists, k_constant=int(settings.retrieval.rrf_k))
        seen: set[str] = set()
        out: list[NodeWithScore] = []
        for n in fused:
            meta = getattr(n.node, "metadata", {}) or {}
            key = str(meta.get("page_id") or n.node.node_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(n)
            if len(out) >= self.top_n:
                break
        return out

    def _emit_final_metrics(
        self,
        nodes: list[NodeWithScore],
        out: list[NodeWithScore],
        *,
        text_nodes: list[NodeWithScore],
        visual_nodes: list[NodeWithScore],
    ) -> None:
        """Emit final rerank telemetry metrics."""
        try:
            configured_k = self.top_n
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
            try:
                scores = [float(n.score or 0.0) for n in out]
            except (TypeError, ValueError):  # pragma: no cover - defensive
                scores = []
            score_mean = float(sum(scores) / len(scores)) if scores else 0.0
            score_max = float(max(scores)) if scores else 0.0
            _emit_rerank_telemetry(
                {
                    "rerank.stage": "final",
                    "rerank.topk": int(configured_k),
                    "rerank.latency_ms": 0,
                    "rerank.timeout": False,
                    "rerank.delta_mrr_at_10": delta_mrr,
                    "rerank.path": path,
                    "rerank.total_timeout_budget_ms": _total_rerank_budget_ms(),
                    "rerank.executor": "owned_thread",
                    "rerank.input_count": len(nodes),
                    "rerank.output_count": len(out),
                    "rerank.score.mean": score_mean,
                    "rerank.score.max": score_max,
                }
            )
        except (RuntimeError, ValueError, OSError, TypeError) as exc:
            redaction = build_pii_log_entry(str(exc), key_id="reranking.metrics")
            logger.warning(
                "Final rerank metrics error; skipping telemetry "
                "(error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )

    def _emit_total_timeout(self, budget_ms: int) -> None:
        """Record exhaustion of the caller-visible total rerank budget."""
        logger.warning("Total rerank budget exhausted; fail-open")
        _emit_rerank_telemetry(
            {
                "rerank.stage": "total",
                "rerank.topk": self.top_n,
                "rerank.latency_ms": budget_ms,
                "rerank.timeout": True,
            }
        )

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
        total_budget_ms = _total_rerank_budget_ms()
        deadline = time.monotonic() + (total_budget_ms / 1000.0)
        text_nodes, visual_nodes = self._split_by_modality(nodes)
        lists: list[list[NodeWithScore]] = []

        try:
            isolated_text_nodes = _isolate_rerank_nodes(text_nodes)
            text_budget_ms = _capped_stage_budget_ms(
                deadline,
                _text_timeout_ms() + 50,
            )
            if text_nodes and text_budget_ms <= 0:
                self._emit_total_timeout(total_budget_ms)
                return nodes
            tr = self._run_text_stage(
                isolated_text_nodes,
                query_bundle,
                timeout_ms=text_budget_ms,
            )
            if tr is None:
                if _remaining_rerank_budget_ms(deadline) <= 0:
                    self._emit_total_timeout(total_budget_ms)
                logger.warning("Text rerank timeout; fail-open")
                return nodes
            if tr:
                lists.append(tr)
        except Exception as exc:
            redaction = build_pii_log_entry(str(exc), key_id="reranking.text.fail_open")
            logger.warning(
                "Text rerank error; fail-open (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            lists.append(list(text_nodes))

        isolated_visual_nodes = _isolate_rerank_nodes(visual_nodes)
        visual_budget_ms = _capped_stage_budget_ms(
            deadline,
            _siglip_timeout_ms(),
        )
        if visual_nodes and visual_budget_ms <= 0:
            self._emit_total_timeout(total_budget_ms)
            return nodes
        lists, timed_out = self._run_visual_stage(
            isolated_visual_nodes,
            query_bundle,
            lists,
            timeout_ms=visual_budget_ms,
        )
        if timed_out:
            if _remaining_rerank_budget_ms(deadline) <= 0:
                self._emit_total_timeout(total_budget_ms)
            return nodes
        if _remaining_rerank_budget_ms(deadline) <= 0:
            self._emit_total_timeout(total_budget_ms)
            out = nodes
        elif not lists:
            out = nodes[: self.top_n]
        else:
            out = self._fuse_and_dedup(lists)
            if _remaining_rerank_budget_ms(deadline) <= 0:
                self._emit_total_timeout(total_budget_ms)
                out = nodes
            else:
                self._emit_final_metrics(
                    nodes,
                    out,
                    text_nodes=text_nodes,
                    visual_nodes=visual_nodes,
                )
        return out

    async def _apostprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Rerank asynchronously without LlamaIndex's default thread pool."""
        if not nodes or not query_bundle:
            return nodes

        total_budget_ms = _total_rerank_budget_ms()
        try:
            async with asyncio.timeout(total_budget_ms / 1000.0):
                text_nodes, visual_nodes = self._split_by_modality(nodes)
                lists: list[list[NodeWithScore]] = []

                try:
                    isolated_text_nodes = _isolate_rerank_nodes(text_nodes)
                    text_result = await self._arun_text_stage(
                        isolated_text_nodes,
                        query_bundle,
                    )
                    if text_result is None:
                        logger.warning("Text rerank timeout; fail-open")
                        return nodes
                    if text_result:
                        lists.append(text_result)
                except Exception as exc:
                    redaction = build_pii_log_entry(
                        str(exc),
                        key_id="reranking.text.fail_open",
                    )
                    logger.warning(
                        "Text rerank error; fail-open (error_type={}, error={})",
                        type(exc).__name__,
                        redaction.redacted,
                    )
                    lists.append(list(text_nodes))

                isolated_visual_nodes = _isolate_rerank_nodes(visual_nodes)
                lists, timed_out = await self._arun_visual_stage(
                    isolated_visual_nodes,
                    query_bundle,
                    lists,
                )
                if timed_out:
                    return nodes
                if not lists:
                    return nodes[: self.top_n]

                out = self._fuse_and_dedup(lists)
                self._emit_final_metrics(
                    nodes,
                    out,
                    text_nodes=text_nodes,
                    visual_nodes=visual_nodes,
                )
                return out
        except TimeoutError:
            self._emit_total_timeout(total_budget_ms)
            return nodes

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


__all__ = [
    "MultimodalReranker",
    "build_text_reranker",
]


# ---- Helpers for factories (DRY) ----
def get_postprocessors(
    mode: str,
    *,
    use_reranking: bool,
    top_n: int | None = None,
) -> list[BaseNodePostprocessor] | None:
    """Return node_postprocessors list for the given mode or None.

    Args:
        mode: One of "vector", "hybrid", or "kg".
        use_reranking: Global toggle; when False returns None.
        top_n: Typed output cap from the router's settings instance.

    Returns:
        List of postprocessors or None when disabled or unavailable.
    """
    if not use_reranking:
        return None
    try:
        if mode in ("vector", "hybrid", "kg"):
            return [MultimodalReranker(top_n=_parse_top_k(top_n))]
    except (ValueError, TypeError):  # defensive: score cast
        return None
    return None
