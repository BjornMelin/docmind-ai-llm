"""Test suite for PropertyGraphIndex configuration (REQ-0049).

Tests LlamaIndex PropertyGraphIndex with domain-specific extractors,
multi-hop traversal, and hybrid vector+graph retrieval.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import Document, NodeWithScore

# These imports will fail initially (TDD approach)
try:
    from src.retrieval.graph.property_graph_config import (
        PropertyGraphConfig,
        create_property_graph_index,
        create_tech_schema,
    )
    from src.retrieval.integration import create_hybrid_retriever
except ImportError:
    # Mock for initial test run
    PropertyGraphConfig = MagicMock
    create_property_graph_index = MagicMock
    create_tech_schema = MagicMock
    create_hybrid_retriever = MagicMock


@pytest.fixture
def sample_documents():
    """Create sample technical documents for graph extraction."""
    docs = [
        Document(
            text="""LlamaIndex is a framework for building LLM applications.
            It uses BGE-M3 for embeddings and integrates with Qdrant for vector storage.
            The framework is optimized for RTX 4090 hardware.""",
            metadata={"source": "doc1.md"},
        ),
        Document(
            text="""DocMind AI uses LlamaIndex for its retrieval pipeline.
            It employs DSPy for query optimization and runs on RTX 4090.
            The system supports PropertyGraphIndex for relationship mapping.""",
            metadata={"source": "doc2.md"},
        ),
        Document(
            text="""BGE-M3 is created by BAAI and supports dense, sparse, and ColBERT 
            embeddings. It integrates seamlessly with LlamaIndex and is part of 
            DocMind AI.""",
            metadata={"source": "doc3.md"},
        ),
    ]
    return docs


@pytest.fixture
def tech_schema():
    """Define domain-specific schema for technical documentation."""
    return {
        "entities": ["FRAMEWORK", "LIBRARY", "MODEL", "HARDWARE", "PERSON", "ORG"],
        "relations": ["USES", "OPTIMIZED_FOR", "PART_OF", "CREATED_BY", "SUPPORTS"],
    }


@pytest.fixture
def mock_llm():
    """Mock LLM for entity extraction."""
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=MagicMock(
            text="Entities: LlamaIndex (FRAMEWORK), BGE-M3 (MODEL), RTX 4090 (HARDWARE)"
        )
    )
    return llm


@pytest.fixture
def mock_vector_store():
    """Mock vector store for hybrid retrieval."""
    store = MagicMock()
    store.query = AsyncMock(
        return_value=[
            NodeWithScore(node=MagicMock(text="Vector result 1"), score=0.9),
            NodeWithScore(node=MagicMock(text="Vector result 2"), score=0.85),
        ]
    )
    return store


@pytest.mark.spec("retrieval-enhancements")
class TestPropertyGraphConfiguration:
    """Test PropertyGraphIndex configuration (REQ-0049)."""

    def test_property_graph_config_creation(self, tech_schema):
        """Test PropertyGraphConfig with domain-specific settings."""
        # This will fail initially - implementation needed
        config = PropertyGraphConfig(
            entities=tech_schema["entities"],
            relations=tech_schema["relations"],
            max_paths_per_chunk=20,
            num_workers=4,
            path_depth=2,
            strict_schema=True,
        )

        assert config.entities == tech_schema["entities"]
        assert config.relations == tech_schema["relations"]
        assert config.path_depth == 2
        assert config.strict_schema is True

    @pytest.mark.asyncio
    async def test_domain_specific_extractors(
        self, sample_documents, tech_schema, mock_llm
    ):
        """Test configuration of SimpleLLMPathExtractor and SchemaLLMPathExtractor."""
        # This will fail initially - implementation needed
        from llama_index.core import Settings

        Settings.llm = mock_llm

        # Create property graph with domain extractors
        index = await create_property_graph_index(
            documents=sample_documents,
            schema=tech_schema,
            llm=mock_llm,
            max_paths_per_chunk=20,
        )

        # Verify extractors are configured
        assert len(index.kg_extractors) >= 2
        assert any("SimpleLLMPathExtractor" in str(e) for e in index.kg_extractors)
        assert any("SchemaLLMPathExtractor" in str(e) for e in index.kg_extractors)

        # Verify schema is applied
        assert index.schema == tech_schema

    @pytest.mark.asyncio
    async def test_entity_extraction_accuracy(
        self, sample_documents, tech_schema, mock_llm
    ):
        """Test entity extraction from technical documentation."""
        # This will fail initially - implementation needed
        index = await create_property_graph_index(
            documents=sample_documents,
            schema=tech_schema,
            llm=mock_llm,
        )

        # Extract entities
        entities = await index.extract_entities(sample_documents[0])

        # Verify expected entities found
        entity_texts = [e["text"] for e in entities]
        assert "LlamaIndex" in entity_texts
        assert "BGE-M3" in entity_texts
        assert "RTX 4090" in entity_texts

        # Verify entity types
        entity_types = [e["type"] for e in entities]
        assert "FRAMEWORK" in entity_types
        assert "MODEL" in entity_types
        assert "HARDWARE" in entity_types

        # Verify confidence scores
        assert all(0.0 <= e["confidence"] <= 1.0 for e in entities)

    @pytest.mark.asyncio
    async def test_relationship_extraction_quality(
        self, sample_documents, tech_schema, mock_llm
    ):
        """Test relationship extraction between entities."""
        # This will fail initially - implementation needed
        index = await create_property_graph_index(
            documents=sample_documents,
            schema=tech_schema,
            llm=mock_llm,
        )

        # Extract relationships
        relationships = await index.extract_relationships(sample_documents[0])

        # Verify expected relationships
        rel_types = [r["type"] for r in relationships]
        assert "USES" in rel_types
        assert "OPTIMIZED_FOR" in rel_types

        # Verify relationship structure
        for rel in relationships:
            assert "source" in rel
            assert "target" in rel
            assert "type" in rel
            assert "confidence" in rel
            assert 0.0 <= rel["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_graph_traversal_performance(
        self, sample_documents, tech_schema, mock_llm
    ):
        """Test multi-hop graph traversal completes within 3 seconds."""
        # This will fail initially - implementation needed
        index = await create_property_graph_index(
            documents=sample_documents,
            schema=tech_schema,
            llm=mock_llm,
        )

        # Build graph from documents
        await index.build_graph(sample_documents)

        # Test 2-hop traversal
        start_time = time.perf_counter()

        query = "How are LlamaIndex and BGE-M3 connected?"
        paths = await index.traverse_graph(
            query=query,
            max_depth=2,
            timeout=3.0,
        )

        elapsed_time = time.perf_counter() - start_time

        assert elapsed_time < 3.0, f"Traversal took {elapsed_time:.2f}s, expected <3s"
        assert len(paths) > 0
        assert all(len(p) <= 3 for p in paths)  # 2 hops = max 3 nodes

    @pytest.mark.asyncio
    async def test_hybrid_vector_graph_retrieval(
        self, sample_documents, tech_schema, mock_llm, mock_vector_store
    ):
        """Test hybrid retrieval combining vector and graph results."""
        # This will fail initially - implementation needed
        index = await create_property_graph_index(
            documents=sample_documents,
            schema=tech_schema,
            llm=mock_llm,
            vector_store=mock_vector_store,
        )

        # Create hybrid retriever
        retriever = create_hybrid_retriever(
            property_graph=index,
            vector_store=mock_vector_store,
            retriever_mode="hybrid",
        )

        # Perform hybrid retrieval
        query = "LlamaIndex BGE-M3 integration"
        results = await retriever.retrieve(query, top_k=10)

        assert len(results) <= 10

        # Verify both vector and graph results included
        result_sources = [r.metadata.get("source") for r in results]
        assert any(s == "vector" for s in result_sources)
        assert any(s == "graph" for s in result_sources)

        # Verify scoring
        assert all(hasattr(r, "score") for r in results)
        assert results[0].score >= results[-1].score  # Sorted by score

    def test_confidence_scoring_for_entities(self, tech_schema):
        """Test confidence scoring for extracted entities."""
        # This will fail initially - implementation needed
        from src.retrieval.graph.property_graph_config import (
            calculate_entity_confidence,
        )

        entity = {
            "text": "LlamaIndex",
            "type": "FRAMEWORK",
            "context": "LlamaIndex is a framework for building LLM applications",
            "extractor": "SchemaLLMPathExtractor",
        }

        confidence = calculate_entity_confidence(entity, tech_schema)

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.8  # High confidence for clear entity

        # Test lower confidence for ambiguous entity
        ambiguous_entity = {
            "text": "system",
            "type": "FRAMEWORK",
            "context": "The system uses various components",
            "extractor": "SimpleLLMPathExtractor",
        }

        ambiguous_confidence = calculate_entity_confidence(
            ambiguous_entity, tech_schema
        )
        assert ambiguous_confidence < confidence

    @pytest.mark.asyncio
    async def test_multi_hop_relationship_traversal(
        self, sample_documents, tech_schema, mock_llm
    ):
        """Test traversing relationships up to depth=2."""
        # This will fail initially - implementation needed
        index = await create_property_graph_index(
            documents=sample_documents,
            schema=tech_schema,
            llm=mock_llm,
            path_depth=2,
        )

        await index.build_graph(sample_documents)

        # Find 2-hop relationships
        start_entity = "DocMind AI"
        relationships = await index.find_relationships(
            entity=start_entity,
            max_hops=2,
        )

        # Verify multi-hop paths found
        assert len(relationships) > 0

        # Check path structure
        for path in relationships:
            assert "nodes" in path
            assert "edges" in path
            assert len(path["nodes"]) <= 3  # Start + 2 hops
            assert len(path["edges"]) <= 2  # 2 relationships max

            # Verify path connectivity
            for i, edge in enumerate(path["edges"]):
                if i == 0:
                    assert edge["source"] == start_entity
                if i < len(path["edges"]) - 1:
                    assert edge["target"] == path["edges"][i + 1]["source"]

    @pytest.mark.asyncio
    async def test_integration_with_existing_qdrant(
        self, sample_documents, tech_schema, mock_llm, mock_vector_store
    ):
        """Test PropertyGraphIndex integrates with existing Qdrant vectorstore."""
        # This will fail initially - implementation needed
        index = await create_property_graph_index(
            documents=sample_documents,
            schema=tech_schema,
            llm=mock_llm,
            vector_store=mock_vector_store,
        )

        # Verify vector store integration
        assert index.vector_store == mock_vector_store

        # Test combined query
        query_engine = index.as_query_engine(
            retriever_mode="hybrid",
            response_mode="tree_summarize",
        )

        response = await query_engine.aquery("Explain LlamaIndex and BGE-M3")

        assert response is not None
        assert hasattr(response, "source_nodes")
        assert len(response.source_nodes) > 0

    @pytest.mark.asyncio
    async def test_performance_with_50_documents(self, tech_schema, mock_llm):
        """Test PropertyGraphIndex performance with realistic document load."""
        # This will fail initially - implementation needed
        # Generate 50 test documents
        large_doc_set = []
        for i in range(50):
            doc = Document(
                text=f"Document {i} discusses Framework_{i % 5} using Model_{i % 3} "
                f"on Hardware_{i % 2}. It's created by Org_{i % 4}.",
                metadata={"id": f"doc_{i}"},
            )
            large_doc_set.append(doc)

        # Build graph with timing
        start_time = time.perf_counter()

        index = await create_property_graph_index(
            documents=large_doc_set,
            schema=tech_schema,
            llm=mock_llm,
            num_workers=4,  # Parallel processing
        )

        await index.build_graph(large_doc_set)

        build_time = time.perf_counter() - start_time

        # Verify performance
        assert build_time < 30.0, f"Building graph took {build_time:.2f}s for 50 docs"

        # Test query performance on large graph
        query_start = time.perf_counter()
        results = await index.query("Framework_0 Model_1 relationship", top_k=5)
        query_time = time.perf_counter() - query_start

        assert query_time < 2.0, f"Query took {query_time:.2f}s on 50-doc graph"
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_incremental_graph_updates(
        self, sample_documents, tech_schema, mock_llm
    ):
        """Test adding new documents to existing PropertyGraphIndex."""
        # This will fail initially - implementation needed
        index = await create_property_graph_index(
            documents=sample_documents[:2],
            schema=tech_schema,
            llm=mock_llm,
        )

        initial_entities = await index.get_entity_count()
        initial_relationships = await index.get_relationship_count()

        # Add new document
        new_doc = Document(
            text="Pytorch is a framework that competes with TensorFlow",
            metadata={"source": "new.md"},
        )

        await index.add_document(new_doc)

        # Verify graph updated
        new_entities = await index.get_entity_count()
        new_relationships = await index.get_relationship_count()

        assert new_entities > initial_entities
        assert new_relationships >= initial_relationships  # May add new relationships

        # Verify new entities accessible
        pytorch_paths = await index.find_entity("Pytorch")
        assert len(pytorch_paths) > 0


@pytest.mark.spec("retrieval-enhancements")
class TestPropertyGraphQueryEngine:
    """Test PropertyGraphIndex query engine functionality."""

    @pytest.mark.asyncio
    async def test_relationship_query_routing(
        self, sample_documents, tech_schema, mock_llm
    ):
        """Test RouterQueryEngine detects and routes relationship queries."""
        # This will fail initially - implementation needed
        from src.retrieval.query_engine.router_engine import RouterQueryEngine

        index = await create_property_graph_index(
            documents=sample_documents,
            schema=tech_schema,
            llm=mock_llm,
        )

        router = RouterQueryEngine()
        router.add_query_engine("property_graph", index.as_query_engine())

        # Test relationship query detection
        relationship_queries = [
            "How are LlamaIndex and BGE-M3 connected?",
            "What is the relationship between DocMind and RTX 4090?",
            "Show connections between frameworks and models",
        ]

        for query in relationship_queries:
            selected_engine = router.select_query_engine(query)
            assert selected_engine.name == "property_graph"

    @pytest.mark.asyncio
    async def test_graph_only_retrieval_mode(
        self, sample_documents, tech_schema, mock_llm
    ):
        """Test graph-only retrieval without vector search."""
        # This will fail initially - implementation needed
        index = await create_property_graph_index(
            documents=sample_documents,
            schema=tech_schema,
            llm=mock_llm,
        )

        # Configure graph-only retriever
        retriever = index.as_retriever(
            include_text=False,  # Graph-only
            similarity_top_k=10,
            path_depth=2,
        )

        results = await retriever.aretrieve("LlamaIndex relationships")

        # Verify only graph results
        assert all(r.metadata.get("source_type") == "graph" for r in results)
        assert all(
            "relationship" in r.metadata or "entity" in r.metadata for r in results
        )
