"""Run BEIR information retrieval evaluation against a Qdrant collection.

This script assumes the BEIR dataset is already downloaded and that indexing
into Qdrant is performed separately. It computes standard IR metrics and
appends the results to a leaderboard CSV with deterministic, explicit fields.

Outputs include dynamic metric headers with explicit ``@{k}`` (lowercase):
``ndcg@{k}``, ``recall@{k}``, ``mrr@{k}`` along with ``schema_version`` and
``sample_count``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Optional imports to keep module importable when BEIR is not installed. Tests
# monkeypatch these symbols on the module to avoid heavy dependencies.
try:  # pragma: no cover - environment-dependent
    from beir.datasets.data_loader import GenericDataLoader  # type: ignore
    from beir.retrieval.evaluation import EvaluateRetrieval  # type: ignore
except Exception:  # pragma: no cover
    GenericDataLoader = None  # type: ignore[assignment]
    EvaluateRetrieval = None  # type: ignore[assignment]

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from src.config import settings
from src.eval.common.determinism import set_determinism
from src.eval.common.io import SCHEMA_VERSION, write_csv_row
from src.eval.common.mapping import build_doc_mapping
from src.retrieval.hybrid import ServerHybridRetriever, _HybridParams


def ensure_collection(client: QdrantClient, name: str) -> None:
    """Ensure a Qdrant collection exists; create it when missing.

    Args:
        client: Qdrant client instance.
        name: Name of the collection to verify or create.
    """
    with contextlib.suppress(Exception):
        client.get_collection(name)
        return
    with contextlib.suppress(Exception):
        client.create_collection(
            collection_name=name,
            vectors_config={
                "text-dense": qm.VectorParams(
                    size=settings.embedding.dimension, distance=qm.Distance.COSINE
                )
            },
            sparse_vectors_config={"text-sparse": qm.SparseVectorParams()},
        )


def main() -> None:
    """Run evaluation for a BEIR dataset and log metrics to CSV."""
    # Determinism first
    set_determinism()
    ap = argparse.ArgumentParser(description="DocMind IR eval on BEIR")
    ap.add_argument(
        "--data_dir", required=True, help="BEIR dataset folder (e.g., scifact)"
    )
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--results_dir", default="eval/results")
    ap.add_argument("--collection", default=settings.database.qdrant_collection)
    ap.add_argument(
        "--sample_count",
        type=int,
        default=0,
        help="If >0, limit number of queries deterministically",
    )
    args = ap.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.sample_count < 0:
        raise ValueError("--sample_count must be >= 0")

    # Import BEIR on-demand if not available at module import time.
    global GenericDataLoader, EvaluateRetrieval
    if GenericDataLoader is None or EvaluateRetrieval is None:  # pragma: no cover
        try:
            from beir.datasets.data_loader import (
                GenericDataLoader as _GenericDataLoader,  # type: ignore
            )
            from beir.retrieval.evaluation import (
                EvaluateRetrieval as _EvaluateRetrieval,  # type: ignore
            )

            GenericDataLoader = _GenericDataLoader  # type: ignore[assignment]
            EvaluateRetrieval = _EvaluateRetrieval  # type: ignore[assignment]
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "beir is required for IR evaluation; install optional eval extras"
            ) from exc

    _corpus, queries, qrels = GenericDataLoader(args.data_dir).load(split="test")

    # Ensure collection exists
    client = QdrantClient(
        url=settings.database.qdrant_url,
        timeout=settings.database.qdrant_timeout,
        prefer_grpc=True,
    )
    ensure_collection(client, args.collection)

    # NOTE: Indexing of BEIR corpus into Qdrant is expected to be done separately.
    retr = ServerHybridRetriever(_HybridParams(collection=args.collection))

    # Optionally limit number of queries for deterministic CI runs
    qitems = list(queries.items())
    if args.sample_count > 0:
        qitems = qitems[: args.sample_count]

    results: dict[str, dict[str, float]] = {}
    per_query_nodes: dict[str, list[Any]] = {}
    for qid, qtext in qitems:
        nodes = retr.retrieve(qtext)
        per_query_nodes[qid] = nodes
        doc_scores: dict[str, float] = {}
        for nws in nodes:
            did = (nws.node.metadata or {}).get("doc_id")
            if did:
                doc_scores[str(did)] = float(nws.score)
        results[qid] = doc_scores

    evaluator = EvaluateRetrieval()
    k_list = [int(args.k)]
    ndcg, _map, recall, _precision = evaluator.evaluate(qrels, results, k_list)
    # MRR via custom metric; support uppercase/lowercase key variants
    mrr_dict = evaluator.evaluate_custom(qrels, results, k_list, metric="mrr")
    mrr_val = mrr_dict.get(f"MRR@{args.k}")
    if mrr_val is None:
        mrr_val = mrr_dict.get(f"mrr@{args.k}", 0.0)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.data_dir).name

    # Build doc mapping for reproducibility
    mapping = build_doc_mapping(per_query_nodes)
    mapping_path = out_dir / "doc_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "ts": datetime.now(UTC).isoformat(),
                "dataset": dataset,
                "k": args.k,
                "sample_count": len(qitems),
                "mapping": mapping,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Leaderboard row with dynamic @k metric headers
    row = {
        "schema_version": SCHEMA_VERSION,
        "ts": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "k": args.k,
        f"ndcg@{args.k}": ndcg.get(f"NDCG@{args.k}", 0.0),
        f"recall@{args.k}": recall.get(f"Recall@{args.k}", 0.0),
        f"mrr@{args.k}": float(mrr_val),
        "sample_count": len(qitems),
    }
    lb = out_dir / "leaderboard.csv"
    write_csv_row(lb, row)


if __name__ == "__main__":
    main()
