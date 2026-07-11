"""Smoke-test Qdrant named-vector RRF and DBSF queries."""

from __future__ import annotations

import argparse
import uuid

from qdrant_client import QdrantClient, models


def run_smoke(location: str) -> None:
    """Create, query, and remove a temporary hybrid collection."""
    client = QdrantClient(location=location, timeout=10)
    collection = f"docmind-fusion-smoke-{uuid.uuid4().hex}"
    created = False
    try:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "text-dense": models.VectorParams(
                    size=2,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(),
            },
        )
        created = True
        client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=1,
                    vector={
                        "text-dense": [1.0, 0.0],
                        "text-sparse": models.SparseVector(indices=[0], values=[1.0]),
                    },
                ),
                models.PointStruct(
                    id=2,
                    vector={
                        "text-dense": [0.8, 0.2],
                        "text-sparse": models.SparseVector(indices=[0], values=[0.5]),
                    },
                ),
                models.PointStruct(
                    id=3,
                    vector={
                        "text-dense": [0.0, 1.0],
                        "text-sparse": models.SparseVector(indices=[1], values=[1.0]),
                    },
                ),
            ],
            wait=True,
        )
        prefetch = [
            models.Prefetch(query=[1.0, 0.0], using="text-dense", limit=3),
            models.Prefetch(
                query=models.SparseVector(indices=[0], values=[1.0]),
                using="text-sparse",
                limit=3,
            ),
        ]
        for fusion in (models.Fusion.RRF, models.Fusion.DBSF):
            result = client.query_points(
                collection_name=collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=fusion),
                limit=3,
            )
            if not result.points:
                raise SystemExit(f"Qdrant {fusion.value} query returned no points")
        print("Qdrant fusion smoke passed: RRF and DBSF")
    finally:
        if created:
            client.delete_collection(collection)
        client.close()


def main() -> None:
    """Parse the Qdrant location and run the fusion smoke."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location",
        default="http://127.0.0.1:6333",
        help="Qdrant URL or :memory: for a local client smoke",
    )
    args = parser.parse_args()
    run_smoke(args.location)


if __name__ == "__main__":
    main()
