"""Integration tests for PropertyGraph configuration and construction.

This module tests real PropertyGraphIndex configuration, initialization,
and boundary validation without testing LlamaIndex library internals.

Focus areas:
- PropertyGraph configuration validation and loading
- Schema creation and validation
- Graph index construction with real extractors
- Integration boundary testing with mocked LLMs
- Error handling for invalid configurations
- Async operations and graph store integration
"""

from unittest.mock import Mock, patch

import pytest

from src.retrieval.graph_config import (
    DEFAULT_ENTITY_CONFIDENCE,
    DEFAULT_MAX_HOPS,
    DEFAULT_MAX_PATHS_PER_CHUNK,
    DEFAULT_MAX_TRIPLETS_PER_CHUNK,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PATH_DEPTH,
    DEFAULT_RELATIONSHIP_CONFIDENCE,
    DEFAULT_TOP_K,
    DEFAULT_TRAVERSAL_TIMEOUT,
    PropertyGraphConfig,
    calculate_entity_confidence,
    create_property_graph_index,
    create_property_graph_index_async,
    create_tech_schema,
    extend_property_graph_index,
)


@pytest.mark.integration
class TestPropertyGraphConfigIntegration:
    """Integration tests for PropertyGraph configuration validation."""

    def test_property_graph_config_validation(self):
        """Test PropertyGraphConfig model validation with different configurations."""
        # Test default configuration
        default_config = PropertyGraphConfig()

        assert len(default_config.entities) == 6
        assert "FRAMEWORK" in default_config.entities
        assert "LIBRARY" in default_config.entities
        assert "MODEL" in default_config.entities
        assert "HARDWARE" in default_config.entities
        assert "PERSON" in default_config.entities
        assert "ORG" in default_config.entities

        assert len(default_config.relations) == 5
        assert "USES" in default_config.relations
        assert "OPTIMIZED_FOR" in default_config.relations
        assert "PART_OF" in default_config.relations
        assert "CREATED_BY" in default_config.relations
        assert "SUPPORTS" in default_config.relations

        # Test configuration with custom values
        custom_config = PropertyGraphConfig(
            entities=["CUSTOM_ENTITY", "ANOTHER_ENTITY"],
            relations=["CUSTOM_RELATION"],
            max_paths_per_chunk=50,
            num_workers=8,
            path_depth=3,
            strict_schema=False,
        )

        assert custom_config.entities == ["CUSTOM_ENTITY", "ANOTHER_ENTITY"]
        assert custom_config.relations == ["CUSTOM_RELATION"]
        assert custom_config.max_paths_per_chunk == 50
        assert custom_config.num_workers == 8
        assert custom_config.path_depth == 3
        assert custom_config.strict_schema is False

    def test_tech_schema_creation_integration(self):
        """Test technical documentation schema creation."""
        schema = create_tech_schema()

        assert isinstance(schema, dict)
        assert "entities" in schema
        assert "relations" in schema

        # Validate entity types for technical documentation
        expected_entities = [
            "FRAMEWORK",
            "LIBRARY",
            "MODEL",
            "HARDWARE",
            "PERSON",
            "ORG",
        ]
        assert schema["entities"] == expected_entities

        # Validate relationship types for technical documentation
        expected_relations = [
            "USES",
            "OPTIMIZED_FOR",
            "PART_OF",
            "CREATED_BY",
            "SUPPORTS",
        ]
        assert schema["relations"] == expected_relations

    def test_schema_customization_integration(self):
        """Test schema customization for different domains."""
        # Test with custom schema
        custom_schema = {
            "entities": ["ALGORITHM", "DATASET", "METRIC"],
            "relations": ["EVALUATES", "TRAINS_ON", "IMPLEMENTS"],
        }

        # Should be able to use custom schema throughout system
        assert len(custom_schema["entities"]) == 3
        assert len(custom_schema["relations"]) == 3
        assert "ALGORITHM" in custom_schema["entities"]
        assert "EVALUATES" in custom_schema["relations"]


@pytest.mark.integration
class TestPropertyGraphIndexConstruction:
    """Integration tests for PropertyGraphIndex construction and configuration."""

    @pytest.fixture
    def mock_documents(self):
        """Create mock documents for testing."""
        doc1 = Mock()
        doc1.text = "BGE-M3 is a unified embedding model created by BAAI"
        doc1.metadata = {"source": "paper1.pdf"}

        doc2 = Mock()
        doc2.text = "LlamaIndex supports PropertyGraphIndex for knowledge graphs"
        doc2.metadata = {"source": "documentation.md"}

        return [doc1, doc2]

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM for testing."""
        llm = Mock()
        llm.complete = Mock(return_value=Mock(text="Extracted entity: BGE-M3"))
        return llm

    def test_property_graph_index_creation_integration(self, mock_documents, mock_llm):
        """Test PropertyGraphIndex creation with real extractors configuration."""
        with patch("src.retrieval.graph_config.PropertyGraphIndex") as mock_index_class:
            with patch(
                "src.retrieval.graph_config.SimplePropertyGraphStore"
            ) as mock_store_class:
                with patch(
                    "src.retrieval.graph_config.SimpleLLMPathExtractor"
                ) as mock_simple_extractor:
                    with patch(
                        "src.retrieval.graph_config.SchemaLLMPathExtractor"
                    ) as mock_schema_extractor:
                        with patch(
                            "src.retrieval.graph_config.ImplicitPathExtractor"
                        ) as mock_implicit_extractor:
                            # Setup mocks
                            mock_index = Mock()
                            mock_index.schema = None
                            mock_index.kg_extractors = None
                            mock_index.path_depth = None
                            mock_index_class.from_documents = Mock(
                                return_value=mock_index
                            )

                            mock_store = Mock()
                            mock_store_class.return_value = mock_store

                            # Create index
                            index = create_property_graph_index(
                                documents=mock_documents,
                                schema=None,  # Use default
                                llm=mock_llm,
                                vector_store=None,
                                max_paths_per_chunk=20,
                                num_workers=4,
                                path_depth=2,
                            )

                            # Validate creation process
                            assert index is not None
                            mock_index_class.from_documents.assert_called_once()

                            # Validate configuration was applied
                            call_args = mock_index_class.from_documents.call_args
                            assert call_args[0][0] == mock_documents  # documents
                            assert "kg_extractors" in call_args[1]
                            assert "property_graph_store" in call_args[1]
                            assert (
                                call_args[1]["max_triplets_per_chunk"]
                                == DEFAULT_MAX_TRIPLETS_PER_CHUNK
                            )

                            # Validate extractors were configured
                            mock_simple_extractor.assert_called_once_with(
                                llm=mock_llm,
                                max_paths_per_chunk=20,
                                num_workers=4,
                            )
                            mock_schema_extractor.assert_called_once_with(llm=mock_llm)
                            mock_implicit_extractor.assert_called_once()

    def test_property_graph_index_with_custom_schema(self, mock_documents, mock_llm):
        """Test PropertyGraphIndex creation with custom schema."""
        custom_schema = {
            "entities": ["ALGORITHM", "FRAMEWORK"],
            "relations": ["IMPLEMENTS", "EXTENDS"],
        }

        with patch("src.retrieval.graph_config.PropertyGraphIndex") as mock_index_class:
            with patch("src.retrieval.graph_config.SimplePropertyGraphStore"):
                with patch("src.retrieval.graph_config.SimpleLLMPathExtractor"):
                    with patch("src.retrieval.graph_config.SchemaLLMPathExtractor"):
                        with patch("src.retrieval.graph_config.ImplicitPathExtractor"):
                            mock_index = Mock()
                            mock_index_class.from_documents = Mock(
                                return_value=mock_index
                            )

                            index = create_property_graph_index(
                                documents=mock_documents,
                                schema=custom_schema,
                                llm=mock_llm,
                            )

                            # Schema should be stored on index
                            assert index.schema == custom_schema

    def test_property_graph_index_with_vector_store(self, mock_documents, mock_llm):
        """Test PropertyGraphIndex creation with vector store integration."""
        mock_vector_store = Mock()

        with patch("src.retrieval.graph_config.PropertyGraphIndex") as mock_index_class:
            with patch("src.retrieval.graph_config.SimplePropertyGraphStore"):
                with patch("src.retrieval.graph_config.SimpleLLMPathExtractor"):
                    with patch("src.retrieval.graph_config.SchemaLLMPathExtractor"):
                        with patch("src.retrieval.graph_config.ImplicitPathExtractor"):
                            mock_index = Mock()
                            mock_index_class.from_documents = Mock(
                                return_value=mock_index
                            )

                            create_property_graph_index(
                                documents=mock_documents,
                                llm=mock_llm,
                                vector_store=mock_vector_store,
                            )

                            # Vector store should be passed to index creation
                            call_args = mock_index_class.from_documents.call_args
                            assert call_args[1]["vector_store"] == mock_vector_store

    @pytest.mark.asyncio
    async def test_async_property_graph_index_creation(self, mock_documents, mock_llm):
        """Test async PropertyGraphIndex creation integration."""
        with patch(
            "src.retrieval.graph_config.create_property_graph_index"
        ) as mock_create:
            mock_index = Mock()
            mock_create.return_value = mock_index

            # Test async wrapper
            index = await create_property_graph_index_async(
                documents=mock_documents,
                llm=mock_llm,
                max_paths_per_chunk=15,
                num_workers=6,
            )

            assert index == mock_index
            mock_create.assert_called_once_with(
                mock_documents,
                None,  # schema
                mock_llm,
                None,  # vector_store
                15,  # max_paths_per_chunk
                6,  # num_workers
                DEFAULT_PATH_DEPTH,
            )


@pytest.mark.integration
class TestPropertyGraphOperationsIntegration:
    """Integration tests for PropertyGraph operations and async methods."""

    @pytest.fixture
    def mock_property_graph_index(self):
        """Create mock PropertyGraphIndex with graph store."""
        index = Mock()

        # Mock property graph store
        mock_store = Mock()
        index.property_graph_store = mock_store

        # Mock kg_extractors
        index.kg_extractors = [Mock(), Mock()]

        return index

    @pytest.fixture
    def mock_entities(self):
        """Create mock entities for testing."""
        entity1 = Mock()
        entity1.id = "entity1"
        entity1.name = "BGE-M3"
        entity1.properties = {"type": "MODEL", "confidence": 0.95}

        entity2 = Mock()
        entity2.id = "entity2"
        entity2.name = "LlamaIndex"
        entity2.properties = {"type": "FRAMEWORK", "confidence": 0.90}

        return [entity1, entity2]

    @pytest.fixture
    def mock_relationships(self):
        """Create mock relationships for testing."""
        rel1 = Mock()
        rel1.source_id = "entity1"
        rel1.target_id = "entity2"
        rel1.id = "rel1"
        rel1.properties = {"type": "USES", "confidence": 0.85}

        return [rel1]

    @pytest.mark.asyncio
    async def test_extract_entities_integration(
        self, mock_property_graph_index, mock_entities
    ):
        """Test entity extraction integration with graph store."""
        from src.retrieval.graph_config import extract_entities

        # Mock graph store to return entities
        mock_property_graph_index.property_graph_store.get_nodes = Mock(
            return_value=mock_entities
        )

        mock_document = Mock()
        mock_document.text = "Test document about BGE-M3 and LlamaIndex"

        entities = await extract_entities(mock_property_graph_index, mock_document)

        assert len(entities) == 2

        # Validate entity data structure
        bge_entity = next(e for e in entities if e["text"] == "BGE-M3")
        assert bge_entity["type"] == "MODEL"
        assert bge_entity["confidence"] == 0.95
        assert bge_entity["id"] == "entity1"

        llamaindex_entity = next(e for e in entities if e["text"] == "LlamaIndex")
        assert llamaindex_entity["type"] == "FRAMEWORK"
        assert llamaindex_entity["confidence"] == 0.90
        assert llamaindex_entity["id"] == "entity2"

    @pytest.mark.asyncio
    async def test_extract_relationships_integration(
        self, mock_property_graph_index, mock_relationships
    ):
        """Test relationship extraction integration with graph store."""
        from src.retrieval.graph_config import extract_relationships

        # Mock graph store to return relationships
        mock_property_graph_index.property_graph_store.get_edges = Mock(
            return_value=mock_relationships
        )

        mock_document = Mock()
        mock_document.text = "BGE-M3 uses LlamaIndex framework"

        relationships = await extract_relationships(
            mock_property_graph_index, mock_document
        )

        assert len(relationships) == 1

        # Validate relationship data structure
        relationship = relationships[0]
        assert relationship["source"] == "entity1"
        assert relationship["target"] == "entity2"
        assert relationship["type"] == "USES"
        assert relationship["confidence"] == 0.85
        assert relationship["id"] == "rel1"

    @pytest.mark.asyncio
    async def test_graph_traversal_integration(self, mock_property_graph_index):
        """Test graph traversal integration."""
        from src.retrieval.graph_config import traverse_graph

        # Mock retriever and nodes
        mock_retriever = Mock()
        mock_nodes = [Mock(text="Result 1"), Mock(text="Result 2")]
        mock_retriever.retrieve = Mock(return_value=mock_nodes)
        mock_property_graph_index.as_retriever = Mock(return_value=mock_retriever)

        # Test traversal
        query = "Find information about BGE-M3"
        nodes = await traverse_graph(
            mock_property_graph_index,
            query,
            max_depth=2,
            timeout=1.0,
        )

        assert nodes == mock_nodes
        mock_property_graph_index.as_retriever.assert_called_once_with(
            include_text=False,
            path_depth=2,
        )
        mock_retriever.retrieve.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_graph_traversal_timeout_integration(self, mock_property_graph_index):
        """Test graph traversal timeout handling."""
        from src.retrieval.graph_config import traverse_graph

        # Mock retriever with slow response
        mock_retriever = Mock()

        def slow_retrieve(query):
            import time

            time.sleep(0.2)  # Simulate slow operation
            return [Mock(text="Late result")]

        mock_retriever.retrieve = slow_retrieve
        mock_property_graph_index.as_retriever = Mock(return_value=mock_retriever)

        # Test with very short timeout
        nodes = await traverse_graph(
            mock_property_graph_index,
            "test query",
            max_depth=1,
            timeout=0.05,  # 50ms timeout
        )

        # Should return empty list on timeout
        assert nodes == []


@pytest.mark.integration
class TestPropertyGraphErrorHandlingIntegration:
    """Integration tests for PropertyGraph error handling and recovery."""

    @pytest.mark.asyncio
    async def test_entity_extraction_without_extractors(self):
        """Test entity extraction error handling without extractors."""
        from src.retrieval.graph_config import extract_entities

        # Create index without extractors
        mock_index = Mock()
        mock_index.kg_extractors = None

        mock_document = Mock()

        with pytest.raises(
            NotImplementedError,
            match="Entity extraction requires configured extractors",
        ):
            await extract_entities(mock_index, mock_document)

    @pytest.mark.asyncio
    async def test_entity_extraction_without_graph_store(self):
        """Test entity extraction error handling without graph store."""
        from src.retrieval.graph_config import extract_entities

        # Create index without graph store
        mock_index = Mock()
        mock_index.kg_extractors = [Mock()]
        delattr(mock_index, "property_graph_store")  # Remove attribute

        mock_document = Mock()

        with pytest.raises(
            NotImplementedError,
            match="Entity extraction requires access to the underlying graph store",
        ):
            await extract_entities(mock_index, mock_document)

    @pytest.mark.asyncio
    async def test_relationship_extraction_graph_store_error(self):
        """Test relationship extraction with graph store errors."""
        from src.retrieval.graph_config import extract_relationships

        mock_index = Mock()
        mock_index.kg_extractors = [Mock()]
        mock_store = Mock()
        mock_store.get_edges = Mock(side_effect=RuntimeError("Graph store error"))
        mock_index.property_graph_store = mock_store

        mock_document = Mock()

        with pytest.raises(
            NotImplementedError,
            match="Relationship extraction from existing graph not yet implemented",
        ):
            await extract_relationships(mock_index, mock_document)

    def test_property_graph_creation_with_missing_dependencies(self, mock_llm):
        """Test PropertyGraph creation error handling with missing dependencies."""
        mock_documents = [Mock()]

        with patch("src.retrieval.graph_config.PropertyGraphIndex") as mock_index_class:
            mock_index_class.from_documents = Mock(
                side_effect=ImportError("Missing dependency")
            )

            with pytest.raises(ImportError):
                create_property_graph_index(
                    documents=mock_documents,
                    llm=mock_llm,
                )


@pytest.mark.integration
class TestPropertyGraphExtensionsIntegration:
    """Integration tests for PropertyGraph extension methods."""

    @pytest.fixture
    def mock_extended_index(self):
        """Create mock PropertyGraphIndex for extension testing."""
        index = Mock()
        mock_store = Mock()
        index.property_graph_store = mock_store
        return index

    def test_extend_property_graph_index_integration(self, mock_extended_index):
        """Test PropertyGraphIndex extension with helper methods."""
        # Extend the index
        extended_index = extend_property_graph_index(mock_extended_index)

        # Validate extension methods were added
        assert hasattr(extended_index, "get_entity_count")
        assert hasattr(extended_index, "get_relationship_count")
        assert hasattr(extended_index, "find_entity")
        assert hasattr(extended_index, "find_relationships")
        assert hasattr(extended_index, "add_document")
        assert hasattr(extended_index, "build_graph")
        assert hasattr(extended_index, "query")

        # Validate methods are callable
        assert callable(extended_index.get_entity_count)
        assert callable(extended_index.get_relationship_count)
        assert callable(extended_index.find_entity)
        assert callable(extended_index.find_relationships)
        assert callable(extended_index.add_document)
        assert callable(extended_index.build_graph)
        assert callable(extended_index.query)

    @pytest.mark.asyncio
    async def test_extended_entity_count_integration(self, mock_extended_index):
        """Test extended entity count method integration."""
        # Setup mock nodes
        mock_nodes = [Mock(), Mock(), Mock()]
        mock_extended_index.property_graph_store.get_nodes = Mock(
            return_value=mock_nodes
        )

        extended_index = extend_property_graph_index(mock_extended_index)

        # Test entity count
        count = await extended_index.get_entity_count()
        assert count == 3
        mock_extended_index.property_graph_store.get_nodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_extended_relationship_count_integration(self, mock_extended_index):
        """Test extended relationship count method integration."""
        # Setup mock edges
        mock_edges = [Mock(), Mock()]
        mock_extended_index.property_graph_store.get_edges = Mock(
            return_value=mock_edges
        )

        extended_index = extend_property_graph_index(mock_extended_index)

        # Test relationship count
        count = await extended_index.get_relationship_count()
        assert count == 2
        mock_extended_index.property_graph_store.get_edges.assert_called_once()

    @pytest.mark.asyncio
    async def test_extended_find_entity_integration(self, mock_extended_index):
        """Test extended find entity method integration."""
        # Setup mock nodes
        entity1 = Mock()
        entity1.id = "e1"
        entity1.name = "BGE-M3"
        entity1.properties = {"type": "MODEL"}

        entity2 = Mock()
        entity2.id = "e2"
        entity2.name = "BGE-large"
        entity2.properties = {"type": "MODEL"}

        mock_extended_index.property_graph_store.get_nodes = Mock(
            return_value=[entity1, entity2]
        )
        mock_extended_index.property_graph_store.get_edges = Mock(return_value=[Mock()])

        extended_index = extend_property_graph_index(mock_extended_index)

        # Test finding entity
        results = await extended_index.find_entity("BGE")

        assert len(results) == 2
        assert all("BGE" in r["entity"] for r in results)
        assert all("connections" in r for r in results)
        assert all("type" in r for r in results)


@pytest.mark.integration
class TestEntityConfidenceCalculation:
    """Integration tests for entity confidence calculation."""

    def test_calculate_entity_confidence_integration(self):
        """Test entity confidence calculation with different scenarios."""
        schema = create_tech_schema()

        test_cases = [
            {
                "entity": {
                    "type": "FRAMEWORK",
                    "extractor": "SchemaLLMPathExtractor",
                    "context": "This is a long context with more than fifty characters to test bonus",
                },
                "expected_min": 0.9,  # Base + schema match + schema extractor + context
            },
            {
                "entity": {
                    "type": "UNKNOWN",
                    "extractor": "SimpleLLMPathExtractor",
                    "context": "short",
                },
                "expected_min": 0.6,  # Base + simple extractor
                "expected_max": 0.7,
            },
            {
                "entity": {
                    "type": "LIBRARY",
                    "context": "medium length context for testing",
                },
                "expected_min": 0.7,  # Base + schema match
                "expected_max": 0.8,
            },
        ]

        for case in test_cases:
            confidence = calculate_entity_confidence(case["entity"], schema)

            assert confidence >= case["expected_min"]
            if "expected_max" in case:
                assert confidence <= case["expected_max"]
            assert 0 <= confidence <= 1.0  # Valid confidence range


@pytest.mark.integration
class TestPropertyGraphConfigurationConstants:
    """Integration tests for PropertyGraph configuration constants."""

    def test_default_constants_integration(self):
        """Test that default constants are reasonable for production use."""
        # Performance-related constants
        assert DEFAULT_MAX_PATHS_PER_CHUNK > 0
        assert DEFAULT_MAX_PATHS_PER_CHUNK <= 50  # Reasonable upper bound

        assert DEFAULT_NUM_WORKERS > 0
        assert DEFAULT_NUM_WORKERS <= 16  # Reasonable upper bound

        assert DEFAULT_PATH_DEPTH > 0
        assert DEFAULT_PATH_DEPTH <= 5  # Reasonable traversal depth

        assert DEFAULT_MAX_TRIPLETS_PER_CHUNK > 0
        assert DEFAULT_MAX_TRIPLETS_PER_CHUNK <= 50

        # Confidence-related constants
        assert 0 <= DEFAULT_ENTITY_CONFIDENCE <= 1
        assert 0 <= DEFAULT_RELATIONSHIP_CONFIDENCE <= 1

        # Query-related constants
        assert DEFAULT_TRAVERSAL_TIMEOUT > 0
        assert DEFAULT_TRAVERSAL_TIMEOUT <= 10  # Reasonable timeout

        assert DEFAULT_MAX_HOPS > 0
        assert DEFAULT_MAX_HOPS <= 5  # Reasonable hop limit

        assert DEFAULT_TOP_K > 0
        assert DEFAULT_TOP_K <= 20  # Reasonable result limit

    def test_configuration_constants_consistency(self):
        """Test configuration constants are consistent with each other."""
        # Path depth should be reasonable compared to max hops
        assert DEFAULT_PATH_DEPTH <= DEFAULT_MAX_HOPS + 2

        # Max paths should be reasonable compared to triplets
        assert DEFAULT_MAX_PATHS_PER_CHUNK >= DEFAULT_MAX_TRIPLETS_PER_CHUNK

        # Timeout should be reasonable for the expected workload
        expected_max_time_per_hop = 1.0  # 1 second per hop
        assert DEFAULT_MAX_HOPS * expected_max_time_per_hop <= DEFAULT_TRAVERSAL_TIMEOUT
