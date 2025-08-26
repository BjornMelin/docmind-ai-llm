#!/usr/bin/env python3
"""Test script to validate BGE-M3 enhancements with library-first methods."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.processing.embeddings.bgem3_embedder import (
    create_bgem3_embedder,
)
from src.processing.embeddings.models import EmbeddingParameters


async def test_enhanced_bgem3_embedder():
    """Test the enhanced BGE-M3 embedder with library-first methods."""
    print("🔧 Testing Enhanced BGE-M3 Embedder with Library-First Methods")
    print("=" * 70)

    # Test data
    queries = ["What is machine learning?", "How does neural network work?"]
    corpus = [
        "Machine learning is a subset of AI that uses statistical techniques.",
        "Neural networks are computing systems inspired by biological neural networks.",
    ]

    try:
        # Test 1: Enhanced initialization with new parameters
        print("\n1. Testing enhanced initialization...")
        embedder = create_bgem3_embedder(
            pooling_method="cls",
            normalize_embeddings=True,
            weights_for_different_modes=[0.5, 0.3, 0.2],
            devices=["cuda"] if sys.platform != "win32" else ["cpu"],
            return_numpy=False,
        )
        print("✅ Enhanced initialization successful")

        # Test 2: Query-optimized encoding
        print("\n2. Testing query-optimized encoding (encode_queries)...")
        query_result = await embedder.encode_queries(queries)
        print(
            f"✅ Query encoding successful: {len(query_result.dense_embeddings)} embeddings"
        )
        print(f"   Model info: {query_result.model_info.get('optimization', 'N/A')}")

        # Test 3: Corpus-optimized encoding
        print("\n3. Testing corpus-optimized encoding (encode_corpus)...")
        corpus_result = await embedder.encode_corpus(corpus)
        print(
            f"✅ Corpus encoding successful: {len(corpus_result.dense_embeddings)} embeddings"
        )
        print(f"   Model info: {corpus_result.model_info.get('optimization', 'N/A')}")

        # Test 4: Unified similarity computation
        print("\n4. Testing unified similarity computation...")
        similarity_scores = embedder.compute_similarity(queries, corpus)
        print("✅ Similarity computation successful")
        print(f"   Available modes: {list(similarity_scores.keys())}")

        # Test 5: Sparse embedding token inspection
        print("\n5. Testing sparse embedding token inspection...")
        sparse_result = await embedder.embed_texts_async(
            queries, EmbeddingParameters(return_sparse=True, return_dense=False)
        )
        if sparse_result.sparse_embeddings:
            tokens = embedder.get_sparse_embedding_tokens(
                sparse_result.sparse_embeddings
            )
            print(f"✅ Token inspection successful: {len(tokens)} token sets")
            if tokens:
                sample_tokens = list(tokens[0].keys())[:5]
                print(f"   Sample tokens: {sample_tokens}")

        # Test 6: Performance stats with library optimization
        print("\n6. Testing performance statistics...")
        stats = embedder.get_performance_stats()
        print("✅ Performance stats:")
        for key, value in stats.items():
            if key in ["pooling_method", "library_optimization", "weights_for_modes"]:
                print(f"   {key}: {value}")

        print("\n🎉 All BGE-M3 library-first enhancements tested successfully!")
        print(f"📊 Total texts processed: {stats['total_texts_embedded']}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(test_enhanced_bgem3_embedder())
    sys.exit(0 if success else 1)
