"""Run BEIR information retrieval evaluation against a Qdrant collection.

This script assumes the BEIR dataset is already downloaded and that indexing
into Qdrant is performed separately. It computes standard IR metrics
(`NDCG@10`, `Recall@10`, `MRR@10`) and appends the results to a CSV leaderboard.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import UTC, datetime
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from src.config import settings
from src.retrieval.query_engine import (
    ServerHybridRetriever,
    _HybridParams,
)


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
    ap = argparse.ArgumentParser(description="DocMind IR eval on BEIR")
    ap.add_argument(
        "--data_dir", required=True, help="BEIR dataset folder (e.g., scifact)"
    )
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--results_dir", default="eval/results")
    ap.add_argument("--collection", default=settings.database.qdrant_collection)
    args = ap.parse_args()

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

    results: dict[str, dict[str, float]] = {}
    for qid, qtext in queries.items():
        nodes = retr.retrieve(qtext)
        doc_scores: dict[str, float] = {}
        for nws in nodes:
            did = (nws.node.metadata or {}).get("doc_id")
            if did:
                doc_scores[str(did)] = float(nws.score)
        results[qid] = doc_scores

    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, _precision = evaluator.evaluate(qrels, results, [10])
    mrr = evaluator.evaluate_custom(qrels, results, [10]).get("mrr@10", 0.0)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.data_dir).name

    # Minimal CSV leaderboard append
    row = {
        "ts": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "k": args.k,
        "ndcg@10": ndcg.get("NDCG@10", 0.0),
        "recall@10": recall.get("Recall@10", 0.0),
        "mrr@10": mrr,
    }
    lb = out_dir / "leaderboard.csv"
    header = not lb.exists()
    with lb.open("a", encoding="utf-8") as f:
        if header:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")


if __name__ == "__main__":
    main()
