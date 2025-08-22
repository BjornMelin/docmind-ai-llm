"""PropertyGraphIndex configuration for REQ-0049.

This module provides LlamaIndex PropertyGraphIndex configuration with
domain-specific extractors for technical documentation.

Library-first implementation using native LlamaIndex features.
"""

import asyncio
from typing import Any

from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SchemaLLMPathExtractor,
    SimpleLLMPathExtractor,
)
from loguru import logger
from pydantic import BaseModel, Field


class PropertyGraphConfig(BaseModel):
    """Configuration for PropertyGraphIndex with domain settings."""

    entities: list[str] = Field(
        default_factory=lambda: [
            "FRAMEWORK",
            "LIBRARY",
            "MODEL",
            "HARDWARE",
            "PERSON",
            "ORG",
        ],
        description="Entity types for extraction",
    )
    relations: list[str] = Field(
        default_factory=lambda: [
            "USES",
            "OPTIMIZED_FOR",
            "PART_OF",
            "CREATED_BY",
            "SUPPORTS",
        ],
        description="Relationship types for extraction",
    )
    max_paths_per_chunk: int = Field(
        default=20,
        description="Maximum paths to extract per chunk",
    )
    num_workers: int = Field(
        default=4,
        description="Number of workers for parallel extraction",
    )
    path_depth: int = Field(
        default=2,
        description="Maximum traversal depth for multi-hop queries",
    )
    strict_schema: bool = Field(
        default=True,
        description="Enforce strict schema validation",
    )


def create_tech_schema() -> dict[str, list[str]]:
    """Create domain-specific schema for technical documentation.

    Returns:
        Dictionary with entity and relation types
    """
    return {
        "entities": ["FRAMEWORK", "LIBRARY", "MODEL", "HARDWARE", "PERSON", "ORG"],
        "relations": ["USES", "OPTIMIZED_FOR", "PART_OF", "CREATED_BY", "SUPPORTS"],
    }


def create_property_graph_index(
    documents: list[Any],
    schema: dict[str, list[str]] | None = None,
    llm: Any | None = None,
    vector_store: Any | None = None,
    max_paths_per_chunk: int = 20,
    num_workers: int = 4,
    path_depth: int = 2,
) -> PropertyGraphIndex:
    """Create PropertyGraphIndex with domain-specific extractors.

    Args:
        documents: Documents to build graph from
        schema: Domain-specific schema for entities/relations
        llm: Language model for extraction
        vector_store: Optional vector store for hybrid retrieval
        max_paths_per_chunk: Max paths per chunk
        num_workers: Parallel workers
        path_depth: Multi-hop traversal depth

    Returns:
        Configured PropertyGraphIndex
    """
    if schema is None:
        schema = create_tech_schema()

    if llm is None:
        llm = Settings.llm

    logger.info("Creating PropertyGraphIndex with domain extractors")

    # Configure extractors - all native LlamaIndex
    kg_extractors = [
        # LLM-based extraction for technical relationships
        SimpleLLMPathExtractor(
            llm=llm,
            max_paths_per_chunk=max_paths_per_chunk,
            num_workers=num_workers,
        ),
        # Schema-guided extraction for consistent entity types
        SchemaLLMPathExtractor(
            llm=llm,
        ),
        # Implicit relationship extraction from structure
        ImplicitPathExtractor(),
    ]

    # Create property graph store
    property_graph_store = SimplePropertyGraphStore()

    # Create index with existing infrastructure
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=kg_extractors,
        property_graph_store=property_graph_store,
        vector_store=vector_store,
        show_progress=True,
        max_triplets_per_chunk=10,
    )

    # Store configuration
    index.schema = schema
    index.kg_extractors = kg_extractors
    index.path_depth = path_depth

    logger.info(f"PropertyGraphIndex created with {len(documents)} documents")
    return index


async def create_property_graph_index_async(
    documents: list[Any],
    schema: dict[str, list[str]] | None = None,
    llm: Any | None = None,
    vector_store: Any | None = None,
    max_paths_per_chunk: int = 20,
    num_workers: int = 4,
    path_depth: int = 2,
) -> PropertyGraphIndex:
    """Async wrapper for creating PropertyGraphIndex.

    Args:
        documents: Documents to build graph from
        schema: Domain-specific schema for entities/relations
        llm: Language model for extraction
        vector_store: Optional vector store for hybrid retrieval
        max_paths_per_chunk: Max paths per chunk
        num_workers: Parallel workers
        path_depth: Multi-hop traversal depth

    Returns:
        Configured PropertyGraphIndex
    """
    import asyncio

    # Run synchronous version in thread executor to avoid event loop conflicts
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        create_property_graph_index,
        documents,
        schema,
        llm,
        vector_store,
        max_paths_per_chunk,
        num_workers,
        path_depth,
    )


async def extract_entities(index: PropertyGraphIndex, document: Any) -> list[dict]:
    """Extract entities from a document using configured extractors.

    Args:
        index: PropertyGraphIndex instance
        document: Document to extract entities from

    Returns:
        List of entities with types and confidence
    """
    if not hasattr(index, "kg_extractors") or not index.kg_extractors:
        raise NotImplementedError(
            "Entity extraction requires configured extractors. "
            "Ensure the PropertyGraphIndex has kg_extractors configured."
        )

    entities = []

    # Extract entities using the graph store's internal representation
    # LlamaIndex extractors work internally during index construction
    # For post-construction entity access, query the graph store directly
    if hasattr(index, "property_graph_store"):
        try:
            # Get all nodes (entities) from the graph store
            all_nodes = await asyncio.to_thread(
                lambda: list(index.property_graph_store.get_nodes())
            )

            for node in all_nodes:
                if hasattr(node, "properties"):
                    entity_data = {
                        "text": getattr(node, "name", str(node.id)),
                        "type": node.properties.get("type", "UNKNOWN"),
                        "confidence": node.properties.get("confidence", 0.8),
                        "id": str(node.id),
                    }
                    entities.append(entity_data)

        except Exception as e:
            logger.warning(f"Could not extract entities from graph store: {e}")
            raise NotImplementedError(
                "Entity extraction from existing graph not yet implemented. "
                "Entities are extracted during index construction."
            ) from e
    else:
        raise NotImplementedError(
            "Entity extraction requires access to the underlying graph store. "
            "Ensure PropertyGraphIndex has a property_graph_store configured."
        )

    return entities


async def extract_relationships(index: PropertyGraphIndex, document: Any) -> list[dict]:
    """Extract relationships from a document using configured extractors.

    Args:
        index: PropertyGraphIndex instance
        document: Document to extract relationships from

    Returns:
        List of relationships with types and confidence
    """
    if not hasattr(index, "kg_extractors") or not index.kg_extractors:
        raise NotImplementedError(
            "Relationship extraction requires configured extractors. "
            "Ensure the PropertyGraphIndex has kg_extractors configured."
        )

    relationships = []

    # Extract relationships using the graph store's internal representation
    # LlamaIndex extractors work internally during index construction
    # For post-construction relationship access, query the graph store directly
    if hasattr(index, "property_graph_store"):
        try:
            # Get all edges (relationships) from the graph store
            all_edges = await asyncio.to_thread(
                lambda: list(index.property_graph_store.get_edges())
            )

            for edge in all_edges:
                if hasattr(edge, "properties"):
                    relationship_data = {
                        "source": str(edge.source_id),
                        "target": str(edge.target_id),
                        "type": edge.properties.get(
                            "type", getattr(edge, "label", "UNKNOWN")
                        ),
                        "confidence": edge.properties.get("confidence", 0.8),
                        "id": str(
                            getattr(edge, "id", f"{edge.source_id}-{edge.target_id}")
                        ),
                    }
                    relationships.append(relationship_data)

        except Exception as e:
            logger.warning(f"Could not extract relationships from graph store: {e}")
            raise NotImplementedError(
                "Relationship extraction from existing graph not yet implemented. "
                "Relationships are extracted during index construction."
            ) from e
    else:
        raise NotImplementedError(
            "Relationship extraction requires access to the underlying graph store. "
            "Ensure PropertyGraphIndex has a property_graph_store configured."
        )

    return relationships


async def traverse_graph(
    index: PropertyGraphIndex,
    query: str,
    max_depth: int = 2,
    timeout: float = 3.0,
) -> list[Any]:
    """Traverse graph with multi-hop queries.

    Args:
        index: PropertyGraphIndex instance
        query: Query string
        max_depth: Maximum traversal depth
        timeout: Timeout in seconds

    Returns:
        List of traversal paths
    """
    # Use the index's built-in retrieval
    retriever = index.as_retriever(
        include_text=False,  # Graph-only
        path_depth=max_depth,
    )

    # Perform retrieval
    import asyncio

    try:
        nodes = await asyncio.wait_for(
            asyncio.to_thread(retriever.retrieve, query),
            timeout=timeout,
        )
        return nodes
    except TimeoutError:
        logger.warning(f"Graph traversal timed out after {timeout}s")
        return []


def calculate_entity_confidence(
    entity: dict[str, Any], schema: dict[str, list[str]]
) -> float:
    """Calculate confidence score for extracted entity.

    Args:
        entity: Entity dictionary
        schema: Schema for validation

    Returns:
        Confidence score between 0 and 1
    """
    confidence = 0.5  # Base confidence

    # Check if entity type is in schema
    if entity.get("type") in schema.get("entities", []):
        confidence += 0.2

    # Check extractor type
    if entity.get("extractor") == "SchemaLLMPathExtractor":
        confidence += 0.2
    elif entity.get("extractor") == "SimpleLLMPathExtractor":
        confidence += 0.1

    # Check context quality
    context = entity.get("context", "")
    if len(context) > 50:
        confidence += 0.1

    return min(confidence, 1.0)


# Extension methods for PropertyGraphIndex
def extend_property_graph_index(index: PropertyGraphIndex) -> PropertyGraphIndex:
    """Add helper methods to PropertyGraphIndex instance.

    Args:
        index: PropertyGraphIndex to extend

    Returns:
        Enhanced PropertyGraphIndex with additional methods
    """
    # Add async wrappers
    index.extract_entities = lambda doc: extract_entities(index, doc)
    index.extract_relationships = lambda doc: extract_relationships(index, doc)
    index.traverse_graph = lambda q, d=2, t=3.0: traverse_graph(index, q, d, t)

    # Add graph statistics methods
    async def get_entity_count() -> int:
        """Get total entity count from the graph store."""
        try:
            if hasattr(index, "property_graph_store"):
                nodes = await asyncio.to_thread(
                    lambda: list(index.property_graph_store.get_nodes())
                )
                return len(nodes)
            else:
                raise NotImplementedError(
                    "Entity count requires access to property_graph_store"
                )
        except Exception as e:
            logger.warning(f"Could not get entity count: {e}")
            raise NotImplementedError(
                "Entity count calculation not yet implemented for this graph store type"
            ) from e

    async def get_relationship_count() -> int:
        """Get total relationship count from the graph store."""
        try:
            if hasattr(index, "property_graph_store"):
                edges = await asyncio.to_thread(
                    lambda: list(index.property_graph_store.get_edges())
                )
                return len(edges)
            else:
                raise NotImplementedError(
                    "Relationship count requires access to property_graph_store"
                )
        except Exception as e:
            logger.warning(f"Could not get relationship count: {e}")
            raise NotImplementedError(
                "Relationship count calculation not yet implemented "
                "for this graph store type"
            ) from e

    async def find_entity(entity_name: str) -> list[dict[str, Any]]:
        """Find specific entity in graph by name."""
        try:
            if hasattr(index, "property_graph_store"):
                # Search for nodes matching the entity name
                all_nodes = await asyncio.to_thread(
                    lambda: list(index.property_graph_store.get_nodes())
                )

                matching_entities = []
                for node in all_nodes:
                    node_name = getattr(node, "name", str(node.id))
                    if entity_name.lower() in node_name.lower():
                        # Count connections for this entity
                        connections = await asyncio.to_thread(
                            lambda node_id=node.id: len(
                                list(
                                    index.property_graph_store.get_edges(
                                        source_ids=[node_id]
                                    )
                                )
                            )
                        )

                        matching_entities.append(
                            {
                                "entity": node_name,
                                "id": str(node.id),
                                "connections": connections,
                                "type": getattr(node, "properties", {}).get(
                                    "type", "UNKNOWN"
                                ),
                            }
                        )

                return matching_entities
            else:
                raise NotImplementedError(
                    "Entity search requires access to property_graph_store"
                )
        except Exception as e:
            logger.warning(f"Could not find entity {entity_name}: {e}")
            raise NotImplementedError(
                "Entity search not yet implemented for this graph store type"
            ) from e

    async def find_relationships(
        entity: str, max_hops: int = 2
    ) -> list[dict[str, Any]]:
        """Find relationships for an entity within max_hops."""
        try:
            if hasattr(index, "property_graph_store"):
                # First find the entity node
                all_nodes = await asyncio.to_thread(
                    lambda: list(index.property_graph_store.get_nodes())
                )

                entity_node = None
                for node in all_nodes:
                    node_name = getattr(node, "name", str(node.id))
                    if entity.lower() in node_name.lower():
                        entity_node = node
                        break

                if not entity_node:
                    return []

                # Initialize relationship traversal
                # Note: Direct edge retrieval not needed for traversal algorithm

                relationships = []
                visited_nodes = {str(entity_node.id)}
                current_nodes = [entity_node.id]

                for hop in range(max_hops):
                    next_nodes = []
                    hop_edges = []

                    for current_node_id in current_nodes:
                        # Use lambda with default parameter to bind variable
                        node_edges = await asyncio.to_thread(
                            lambda node_id=current_node_id: list(
                                index.property_graph_store.get_edges(
                                    source_ids=[node_id]
                                )
                            )
                        )

                        for edge in node_edges:
                            if str(edge.target_id) not in visited_nodes:
                                hop_edges.append(
                                    {
                                        "source": str(edge.source_id),
                                        "target": str(edge.target_id),
                                        "type": edge.properties.get(
                                            "type", getattr(edge, "label", "UNKNOWN")
                                        ),
                                        "hop": hop + 1,
                                    }
                                )
                                next_nodes.append(edge.target_id)
                                visited_nodes.add(str(edge.target_id))

                    if hop_edges:
                        relationships.append(
                            {
                                "nodes": list(visited_nodes),
                                "edges": hop_edges,
                                "hop_level": hop + 1,
                            }
                        )

                    current_nodes = next_nodes
                    if not current_nodes:
                        break

                return relationships
            else:
                raise NotImplementedError(
                    "Relationship traversal requires access to property_graph_store"
                )
        except Exception as e:
            logger.warning(f"Could not find relationships for {entity}: {e}")
            raise NotImplementedError(
                "Relationship traversal not yet implemented for this graph store type"
            ) from e

    async def add_document(document: Any) -> None:
        """Add new document to existing graph."""
        # The index handles this internally
        index.insert(document)

    async def build_graph(documents: list[Any]) -> None:
        """Build graph from documents."""
        # The index handles this internally
        for doc in documents:
            index.insert(doc)

    async def query(query_str: str, top_k: int = 5) -> list[Any]:
        """Query the property graph."""
        retriever = index.as_retriever(similarity_top_k=top_k)
        return await asyncio.to_thread(retriever.retrieve, query_str)

    # Attach methods
    index.get_entity_count = get_entity_count
    index.get_relationship_count = get_relationship_count
    index.find_entity = find_entity
    index.find_relationships = find_relationships
    index.add_document = add_document
    index.build_graph = build_graph
    index.query = query

    return index
