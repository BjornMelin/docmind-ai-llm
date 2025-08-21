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
    top_n: int = Field(default=5)
    device: str = Field(default="cuda")
    use_fp16: bool = Field(default=True)
    normalize_scores: bool = Field(default=True)
    batch_size: int = Field(default=16)
    max_length: int = Field(default=512)

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 5,
        device: str = "cuda",
        use_fp16: bool = True,
        normalize_scores: bool = True,
        batch_size: int = 16,
        max_length: int = 512,
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

            logger.info(f"BGE CrossEncoder loaded: {model_name} (FP16: {use_fp16})")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder: {e}")
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

        if len(nodes) <= 1:
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

            # Apply score normalization if requested
            if self.normalize_scores:
                scores = torch.sigmoid(torch.tensor(scores)).numpy()

            # Update node scores and create reranked list
            reranked_nodes = []
            for node, score in zip(nodes, scores, strict=False):
                # Update node with CrossEncoder relevance score
                node.score = float(score)
                reranked_nodes.append(node)

            # Sort by relevance score (descending order)
            reranked_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)

            # Return top_n results
            result_nodes = reranked_nodes[: self.top_n]

            logger.debug(f"Reranked {len(nodes)} nodes -> top {len(result_nodes)}")
            return result_nodes

        except Exception as e:
            logger.error(f"CrossEncoder reranking failed: {e}")
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
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_n: int = 5,
    use_fp16: bool = True,
    device: str = "cuda",
    batch_size: int = 16,
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
    # RTX 4090 optimized batch size
    batch_size = max(batch_size, 16) if device == "cuda" else min(batch_size, 4)

    return BGECrossEncoderRerank(
        model_name=model_name,
        top_n=top_n,
        device=device,
        use_fp16=use_fp16,
        normalize_scores=True,
        batch_size=batch_size,
        max_length=512,  # Optimal for BGE-reranker-v2-m3
    )


# Performance monitoring helper
def benchmark_reranking_latency(
    reranker: BGECrossEncoderRerank, query: str, documents: list[str], num_runs: int = 5
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
    reranker._postprocess_nodes(nodes[:3], query_bundle)

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        reranker._postprocess_nodes(nodes, query_bundle)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return {
        "mean_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "num_documents": len(documents),
        "target_latency_ms": 100.0,  # RTX 4090 target
    }
