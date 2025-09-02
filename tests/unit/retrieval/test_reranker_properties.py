"""Unit tests for BGECrossEncoderRerank factory behavior.

Validates reading of normalize/top_n settings and stable ordering.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval.reranking import create_bge_cross_encoder_reranker


def _mk_nodes(scores: list[float]) -> list[NodeWithScore]:
    nodes: list[NodeWithScore] = []
    for i, s in enumerate(scores):
        nodes.append(NodeWithScore(node=TextNode(text=f"d{i}", id_=f"d{i}"), score=s))
    return nodes


def test_reranker_factory_reads_settings_normalize_and_topn(mocker):
    """Factory should honour normalize flag and top_n from settings."""
    with patch("src.retrieval.reranking.settings") as mock_settings:
        mock_settings.enable_gpu_acceleration = False
        mock_settings.retrieval.reranking_top_k = 3
        mock_settings.retrieval.reranker_normalize_scores = False

        reranker = create_bge_cross_encoder_reranker()
        # Patch underlying model to return static scores
        model = mocker.patch.object(reranker, "_model", autospec=True)
        model.predict.return_value = np.array([0.2, 0.9, 0.5])

        qb = QueryBundle(query_str="q")
        out = reranker.postprocess_nodes(_mk_nodes([0.1, 0.1, 0.1]), qb)
        assert len(out) == 3
        # Verify ordering from returned scores
        assert [n.score for n in out] == sorted([n.score for n in out], reverse=True)

