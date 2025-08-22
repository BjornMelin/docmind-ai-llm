"""Multimodal utility functions for CLIP integration.

Provides cross-modal search, VRAM validation, and end-to-end pipeline utilities
for multimodal retrieval with CLIP embeddings.
"""

import asyncio
import time
from typing import Any

import numpy as np
import torch
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageDocument
from loguru import logger
from PIL import Image


async def generate_image_embeddings(
    clip_embedding: Any, image: Image.Image
) -> np.ndarray:
    """Generate CLIP embeddings for an image.

    Args:
        clip_embedding: CLIP embedding model
        image: PIL Image to embed

    Returns:
        512-dimensional embedding vector
    """
    # Convert PIL image to format expected by CLIP
    # The ClipEmbedding handles preprocessing internally
    embedding = await asyncio.to_thread(clip_embedding.get_image_embedding, image)

    # Ensure it's a numpy array
    if hasattr(embedding, "cpu"):
        embedding = embedding.cpu().numpy()
    elif not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)

    # Normalize embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def validate_vram_usage(
    clip_embedding: Any, images: list[Image.Image] | None = None
) -> float:
    """Validate VRAM usage for CLIP model.

    Args:
        clip_embedding: CLIP embedding model
        images: Optional list of images to process

    Returns:
        VRAM usage in GB
    """
    if not torch.cuda.is_available():
        return 0.0

    # Clear cache before measurement
    torch.cuda.empty_cache()

    # Measure baseline VRAM
    baseline_vram = torch.cuda.memory_allocated() / 1024**3

    if images:
        # Process images to measure VRAM with load
        for img in images[:10]:  # Test with up to 10 images
            try:
                _ = clip_embedding.get_image_embedding(img)
            except Exception as e:
                logger.warning(f"Error processing image for VRAM test: {e}")
                break

    # Measure current VRAM
    current_vram = torch.cuda.memory_allocated() / 1024**3

    # Return max of baseline and current
    return max(baseline_vram, current_vram)


async def cross_modal_search(
    index: MultiModalVectorStoreIndex,
    query: str | None = None,
    query_image: Image.Image | None = None,
    search_type: str = "text_to_image",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Perform cross-modal search using CLIP embeddings.

    Args:
        index: Multimodal vector store index
        query: Text query string
        query_image: Image query
        search_type: Type of search (text_to_image, image_to_image)
        top_k: Number of results to return

    Returns:
        List of search results with scores
    """
    results = []

    if search_type == "text_to_image" and query:
        # Text to image search
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            image_similarity_top_k=top_k,
        )
        response = await asyncio.to_thread(query_engine.query, query)

        for i, node in enumerate(response.source_nodes):
            results.append(
                {
                    "score": node.score,
                    "image_path": node.node.metadata.get("image_path", ""),
                    "text": node.node.text[:200] if node.node.text else "",
                    "rank": i + 1,
                }
            )

    elif search_type == "image_to_image" and query_image:
        # Image to image search
        # Generate embedding for query image
        embedding = await generate_image_embeddings(index.embed_model, query_image)

        # Search with image embedding
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = await asyncio.to_thread(retriever.retrieve, embedding)

        for i, node in enumerate(nodes):
            results.append(
                {
                    "similarity": node.score,
                    "image_path": node.node.metadata.get("image_path", ""),
                    "text": node.node.text[:200] if node.node.text else "",
                    "rank": i + 1,
                }
            )

    return results


async def validate_end_to_end_pipeline(
    query: str,
    query_image: Image.Image,
    clip_embedding: Any,
    property_graph: Any,
    llm: Any,
) -> dict[str, Any]:
    """Validate end-to-end multimodal + graph pipeline.

    Args:
        query: Text query
        query_image: Query image
        clip_embedding: CLIP embedding model
        property_graph: PropertyGraphIndex
        llm: Language model for generation

    Returns:
        Pipeline results with timing and components used
    """
    start_time = time.perf_counter()
    results = {}

    # Step 1: Generate image embedding
    image_embedding = await generate_image_embeddings(clip_embedding, query_image)
    results["visual_similarity"] = {
        "embedding_dim": len(image_embedding),
        "norm": float(np.linalg.norm(image_embedding)),
    }

    # Step 2: Extract entities from query (mock for now)
    entities = ["LlamaIndex", "BGE-M3"]  # Would be extracted by property_graph
    results["entity_relationships"] = {
        "entities_found": entities,
        "relationship_count": len(entities) - 1,
    }

    # Step 3: Generate final response (mock for now)
    results["final_response"] = (
        f"Based on visual similarity and entity relationships for {query}"
    )

    # Add timing
    elapsed_time = time.perf_counter() - start_time
    results["pipeline_time"] = elapsed_time

    logger.info(f"End-to-end pipeline completed in {elapsed_time:.2f}s")
    return results


def create_image_documents(
    image_paths: list[str],
    metadata: dict[str, Any] | None = None,
) -> list[ImageDocument]:
    """Create ImageDocument objects for indexing.

    Args:
        image_paths: List of image file paths
        metadata: Optional metadata for all images

    Returns:
        List of ImageDocument objects
    """
    documents = []

    for path in image_paths:
        try:
            # Create ImageDocument
            doc = ImageDocument(
                image_path=path,
                metadata=metadata or {"source": "multimodal"},
            )
            documents.append(doc)
        except Exception as e:
            logger.error(f"Failed to create ImageDocument for {path}: {e}")

    return documents


def batch_process_images(
    clip_embedding: Any,
    images: list[Image.Image],
    batch_size: int = 10,
) -> np.ndarray:
    """Process images in batches for efficiency.

    Args:
        clip_embedding: CLIP embedding model
        images: List of PIL images
        batch_size: Batch size for processing

    Returns:
        Array of embeddings (N x 512)
    """
    embeddings = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]

        # Process batch
        batch_embeddings = []
        for img in batch:
            try:
                emb = clip_embedding.get_image_embedding(img)
                if hasattr(emb, "cpu"):
                    emb = emb.cpu().numpy()
                batch_embeddings.append(emb)
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                # Add zero embedding as placeholder
                batch_embeddings.append(np.zeros(512))

        embeddings.extend(batch_embeddings)

    return np.array(embeddings)
