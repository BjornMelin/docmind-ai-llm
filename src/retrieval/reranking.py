"""CrossEncoder reranking implementation with BGE-reranker-v2-m3.

This module implements the complete architectural replacement of ColbertRerank
with sentence-transformers CrossEncoder per ADR-006, providing superior
relevance scoring and result ordering.

Key features:
- BGE-reranker-v2-m3 model with 568M parameters
- Direct query-document relevance scoring
- Multilingual support (100+ languages)
- FP16 acceleration for RTX 4090 optimization
- Configurable score normalization
"""

from typing import Any

import numpy as np
import torch
from loguru import logger

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    logger.error(
        "sentence-transformers not available. Install with: "
        "uv add sentence-transformers"
    )
    CrossEncoder = None

from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from src.config.settings import settings

# Cross-Encoder Reranking Constants
DEFAULT_TOP_N = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 512
CUDA_BATCH_SIZE_MIN = 16
CPU_BATCH_SIZE_MAX = 4
WARMUP_SIZE = 3
MS_CONVERSION_FACTOR = 1000
RTX_4090_TARGET_LATENCY_MS = 100.0
SINGLE_NODE_THRESHOLD = 1


class BGECrossEncoderRerank(BaseNodePostprocessor):
    """CrossEncoder reranker using BGE-reranker-v2-m3.

    Advanced reranking using sentence-transformers CrossEncoder with
    BGE-reranker-v2-m3 model for improved result relevance and ordering.
    Replaces ColbertRerank with modern cross-encoder architecture per ADR-006.

    Features:
    - Direct query-document relevance scoring
    - Multilingual support (100+ languages)
    - 568M parameter model for high accuracy
    - GPU acceleration with FP16 support
    - Configurable score normalization
    - Batch processing for efficiency

    Performance targets (RTX 4090 Laptop):
    - <100ms reranking latency for 20 documents
    - <1.2GB VRAM usage
    - >10% NDCG improvement vs no reranking
    """

    model_name: str = Field(default="BAAI/bge-reranker-v2-m3")
    top_n: int = Field(default=DEFAULT_TOP_N)
    device: str = Field(default="cuda")
    use_fp16: bool = Field(default=True)
    normalize_scores: bool = Field(default=True)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE)
    max_length: int = Field(default=DEFAULT_MAX_LENGTH)

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = DEFAULT_TOP_N,
        device: str = "cuda",
        use_fp16: bool = True,
        normalize_scores: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        **kwargs,
    ):
        """Initialize BGE CrossEncoder reranker.

        Args:
            model_name: BGE reranker model identifier
            top_n: Number of top results to return after reranking
            device: Target device (cuda/cpu)
            use_fp16: Enable FP16 acceleration
            normalize_scores: Apply sigmoid normalization to scores
            batch_size: Batch size for RTX 4090 optimization
            max_length: Maximum input length for CrossEncoder
            **kwargs: Additional BaseNodePostprocessor arguments
        """
        super().__init__(
            model_name=model_name,
            top_n=top_n,
            device=device,
            use_fp16=use_fp16,
            normalize_scores=normalize_scores,
            batch_size=batch_size,
            max_length=max_length,
            **kwargs,
        )

        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers not available. Install with: "
                "uv add sentence-transformers"
            )

        try:
            self._model = CrossEncoder(
                model_name, device=device, trust_remote_code=True, max_length=max_length
            )

            # Enable FP16 if requested and supported
            if use_fp16 and torch.cuda.is_available():
                self._model.model.half()
                logger.info("CrossEncoder using FP16 acceleration")

            logger.info("BGE CrossEncoder loaded: %s (FP16: %s)", model_name, use_fp16)
        except (ImportError, RuntimeError, OSError) as e:
            logger.error("Failed to load CrossEncoder: %s", e)
            raise

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Rerank nodes using CrossEncoder scores.

        Core reranking logic that scores query-document pairs and
        returns top-k results ordered by relevance.

        Args:
            nodes: List of nodes with initial scores
            query_bundle: Query information including text

        Returns:
            Reranked and truncated list of nodes with updated scores
        """
        if not query_bundle or not nodes:
            return nodes

        if len(nodes) <= SINGLE_NODE_THRESHOLD:
            return nodes[: self.top_n]

        query_text = query_bundle.query_str

        try:
            # Prepare query-document pairs for CrossEncoder
            pairs = []
            for node in nodes:
                pairs.append([query_text, node.node.get_content()])

            # Compute relevance scores in batches
            scores = self._model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False
            )

            # Apply score normalization if requested, avoiding double normalization
            if self.normalize_scores:
                np_scores = np.asarray(scores)
                # If scores already appear to be probabilities in [0,1],
                # skip extra sigmoid to avoid double-normalization.
                if not (
                    np_scores.size > 0
                    and np_scores.min() >= -1e-6
                    and np_scores.max() <= 1.0 + 1e-6
                ):
                    scores = torch.sigmoid(torch.tensor(np_scores)).numpy()
                else:
                    scores = np_scores

            # Update node scores and create reranked list
            reranked_nodes = []
            for node, score in zip(nodes, scores, strict=False):
                # Create a fresh NodeWithScore to avoid mutating inputs across calls
                reranked_nodes.append(NodeWithScore(node=node.node, score=float(score)))

            # Sort by relevance score (descending order)
            reranked_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)

            # Return top_n results
            result_nodes = reranked_nodes[: self.top_n]

            logger.debug("Reranked %d nodes -> top %d", len(nodes), len(result_nodes))
            return result_nodes

        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
            logger.error("CrossEncoder reranking failed: %s", e)
            # Fallback: return original nodes truncated to top_n
            logger.info("Falling back to original node ordering")
            return nodes[: self.top_n]

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded CrossEncoder model.

        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "normalize_scores": self.normalize_scores,
            "top_n": self.top_n,
        }


def create_bge_cross_encoder_reranker(
    model_name: str | None = None,
    top_n: int | None = None,
    use_fp16: bool | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> BGECrossEncoderRerank:
    """Create BGE CrossEncoder reranker with optimal settings for RTX 4090.

    Factory function following library-first principle for easy instantiation
    with performance-optimized defaults.

    Args:
        model_name: BGE reranker model identifier
        top_n: Number of top results to return
        use_fp16: Enable FP16 acceleration
        device: Target device (cuda/cpu)
        batch_size: Batch size for RTX 4090 optimization

    Returns:
        Configured BGECrossEncoderRerank instance optimized for RTX 4090 Laptop
    """
    # Resolve from settings when available
    # Resolve parameters with precedence: explicit args > settings > defaults
    try:
        cfg = settings.retrieval
        resolved_top_n = top_n if top_n is not None else int(getattr(cfg, "reranking_top_k", DEFAULT_TOP_N))
        normalize = bool(getattr(cfg, "reranker_normalize_scores", True))
        resolved_device = device if device is not None else ("cuda" if getattr(settings, "enable_gpu_acceleration", True) else "cpu")
    except Exception:  # noqa: BLE001
        resolved_top_n = top_n if top_n is not None else DEFAULT_TOP_N
        normalize = True
        resolved_device = device if device is not None else "cuda"

    resolved_model = model_name or "BAAI/bge-reranker-v2-m3"
    resolved_use_fp16 = use_fp16 if use_fp16 is not None else True
    resolved_batch = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE

    # RTX 4090 optimized batch size / CPU conservative
    batch = (
        max(resolved_batch, CUDA_BATCH_SIZE_MIN)
        if resolved_device == "cuda"
        else min(resolved_batch, CPU_BATCH_SIZE_MAX)
    )

    return BGECrossEncoderRerank(
        model_name=resolved_model,
        top_n=resolved_top_n,
        device=resolved_device,
        use_fp16=resolved_use_fp16,
        normalize_scores=normalize,
        batch_size=batch,
        max_length=DEFAULT_MAX_LENGTH,  # Optimal for BGE-reranker-v2-m3
    )


def benchmark_reranking_latency(
    reranker: BGECrossEncoderRerank,
    query: str,
    documents: list[str],
    num_runs: int = DEFAULT_TOP_N,
) -> dict[str, float]:
    """Benchmark reranking latency for performance validation.

    Measures reranking performance to validate <100ms target on RTX 4090.

    Args:
        reranker: BGECrossEncoderRerank instance
        query: Test query
        documents: List of test documents
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with latency statistics
    """
    import time

    from llama_index.core.schema import TextNode

    # Create test nodes
    nodes = []
    for i, doc in enumerate(documents):
        node = NodeWithScore(node=TextNode(text=doc, id_=f"test_node_{i}"), score=1.0)
        nodes.append(node)

    # Create query bundle
    query_bundle = QueryBundle(query_str=query)

    # Warm up
    reranker.postprocess_nodes(nodes[:WARMUP_SIZE], query_bundle)

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        reranker.postprocess_nodes(nodes, query_bundle)
        end_time = time.perf_counter()
        latencies.append(
            (end_time - start_time) * MS_CONVERSION_FACTOR
        )  # Convert to ms

    return {
        "mean_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "num_documents": len(documents),
        "target_latency_ms": RTX_4090_TARGET_LATENCY_MS,  # RTX 4090 target
    }
