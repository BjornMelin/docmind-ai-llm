# Testing Patterns and Best Practices

## Executive Summary

This document defines the specific testing patterns implemented in DocMind AI following the successful test refactoring that achieved **77.8% mock reduction** (21 → 6 @patch decorators). These patterns emphasize **boundary testing**, **library-first approaches**, and **realistic test data** for maintainable, reliable tests.

## Core Testing Patterns

### 1. Boundary Testing Pattern

**Principle**: Mock at system boundaries, not implementation details.

#### System Resource Boundary Pattern

Replace multiple system-level patches with a single comprehensive fixture:

```python
@pytest.fixture
def system_resource_boundary():
    """
    Boundary testing fixture for system resource operations.
    Replaces: @patch("psutil.Process"), @patch("psutil.cpu_percent"), etc.
    """
    # Realistic system resource data
    memory_info = Mock()
    memory_info.rss = 100 * 1024 * 1024  # 100MB
    memory_info.percent = 8.5
    
    virtual_memory = Mock() 
    virtual_memory.percent = 65.0
    virtual_memory.available = 2 * 1024 * 1024 * 1024  # 2GB
    virtual_memory.total = 8 * 1024 * 1024 * 1024      # 8GB
    
    disk_usage = Mock()
    disk_usage.percent = 45.0
    disk_usage.free = 500 * 1024 * 1024 * 1024   # 500GB
    disk_usage.total = 1024 * 1024 * 1024 * 1024  # 1TB
    
    mock_process = Mock()
    mock_process.memory_info.return_value = memory_info
    mock_process.cpu_percent.return_value = 25.0
    
    with (
        patch("psutil.Process", return_value=mock_process),
        patch("psutil.cpu_percent", return_value=35.5),
        patch("psutil.virtual_memory", return_value=virtual_memory),
        patch("psutil.disk_usage", return_value=disk_usage),
        patch("psutil.getloadavg", return_value=(1.2, 1.5, 1.1), create=True),
    ):
        yield {
            "process": mock_process,
            "memory_info": memory_info,
            "virtual_memory": virtual_memory,
            "disk_usage": disk_usage,
            "expected_memory_mb": 100,
            "expected_cpu_percent": 35.5,
            "expected_memory_percent": 65.0
        }
```

**Usage Example**:

```python
@pytest.mark.unit
def test_system_monitoring_logic(system_resource_boundary):
    """Test system monitoring business logic with boundary fixture."""
    monitor = SystemMonitor()
    info = monitor.get_system_info()
    
    # Test business logic outcomes
    assert info["memory_mb"] == 100
    assert info["cpu_percent"] == 35.5
    assert info["memory_percent"] == 65.0
    assert info["status"] == "healthy"
    
    # Verify system calls were made correctly
    boundary = system_resource_boundary
    boundary["process"].memory_info.assert_called_once()
```

**Benefits**:
- ✅ Reduces 5+ @patch decorators to 1 fixture
- ✅ Provides realistic, structured data
- ✅ Reusable across multiple tests
- ✅ Easy to maintain and modify

#### Performance Timing Boundary Pattern

Replace timing-related patches with deterministic performance fixture:

```python
@pytest.fixture
def performance_boundary():
    """
    Deterministic performance testing boundary.
    Replaces: @patch('time.perf_counter', side_effect=[...])
    """
    # Deterministic timing sequence for predictable tests
    timing_sequence = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5]
    
    with patch('time.perf_counter', side_effect=timing_sequence):
        yield {
            'timing_sequence': timing_sequence,
            'expected_durations': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            'total_expected_time': 10.5,
        }
```

**Usage Example**:

```python
@pytest.mark.unit
def test_performance_timer_accuracy(performance_boundary):
    """Test performance timer calculation logic."""
    timer = PerformanceTimer()
    
    timer.start("operation_1")
    timer.end("operation_1")
    
    timer.start("operation_2") 
    timer.end("operation_2")
    
    metrics = timer.get_metrics()
    
    # Test calculation accuracy
    assert metrics["operation_1"]["duration"] == 1.5
    assert metrics["operation_2"]["duration"] == 1.5
    assert metrics["total_operations"] == 2
```

#### Logging Boundary Pattern

Centralize logging verification with structured helper methods:

```python
@pytest.fixture  
def logging_boundary():
    """
    Structured logging testing boundary.
    Replaces: @patch('src.utils.monitoring.logger')
    """
    mock_logger = Mock()
    
    # Helper methods for common assertions
    def assert_info_called(message_contains=None):
        mock_logger.info.assert_called()
        if message_contains:
            call_args = mock_logger.info.call_args[0][0]
            assert message_contains in call_args
    
    def assert_error_called(message_contains=None):
        mock_logger.error.assert_called()
        if message_contains:
            call_args = mock_logger.error.call_args[0][0]
            assert message_contains in call_args
    
    def assert_warning_called(message_contains=None):
        mock_logger.warning.assert_called()
        if message_contains:
            call_args = mock_logger.warning.call_args[0][0]
            assert message_contains in call_args
    
    with patch('src.utils.monitoring.logger', mock_logger):
        yield {
            'logger': mock_logger,
            'assert_info_called': assert_info_called,
            'assert_error_called': assert_error_called,
            'assert_warning_called': assert_warning_called,
            'call_count': lambda level: getattr(mock_logger, level).call_count,
        }
```

**Usage Example**:

```python
@pytest.mark.unit
def test_error_logging_behavior(logging_boundary):
    """Test error logging with structured boundary fixture."""
    processor = DocumentProcessor()
    
    # Trigger error condition
    result = processor.process_invalid_document("")
    
    # Verify business outcome
    assert result.status == "failed"
    assert result.error_type == "validation_error"
    
    # Verify logging behavior
    logging_boundary['assert_error_called']("Document content cannot be empty")
    assert logging_boundary['call_count']('error') == 1
```

### 2. AI Stack Boundary Pattern

**Pattern**: Use LlamaIndex built-in mocks for AI/ML component testing.

#### AI Component Boundary

```python
@pytest.fixture
def ai_stack_boundary():
    """
    Boundary testing for AI/ML components using LlamaIndex mocks.
    Replaces: Multiple AI-related patches with proper mock components.
    """
    from llama_index.core import Settings
    from llama_index.core.llms import MockLLM
    from llama_index.core.embeddings import MockEmbedding
    
    # Use official LlamaIndex mocks for consistency
    mock_llm = MockLLM(max_tokens=256, temperature=0.0)
    mock_embed_model = MockEmbedding(embed_dim=1024)
    
    # Configure global settings
    original_llm = Settings.llm
    original_embed_model = Settings.embed_model
    
    Settings.llm = mock_llm
    Settings.embed_model = mock_embed_model
    
    yield {
        'llm': mock_llm,
        'embed_model': mock_embed_model,
        'embedding_dim': 1024,
        'max_tokens': 256,
        'expected_embedding_length': 1024,
        'test_embedding': [0.1] * 1024,  # Predictable test embedding
        'test_llm_response': "This is a mocked LLM response for testing."
    }
    
    # Cleanup
    Settings.llm = original_llm
    Settings.embed_model = original_embed_model
```

**Usage Example**:

```python
@pytest.mark.unit
@pytest.mark.agents
def test_embedding_pipeline_logic(ai_stack_boundary):
    """Test embedding pipeline business logic with AI boundary."""
    pipeline = EmbeddingPipeline()
    
    # Test with known inputs
    text = "AI research document"
    embedding = pipeline.embed_text(text)
    
    # Test business logic
    assert len(embedding) == ai_stack_boundary['embedding_dim']
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)
    
    # Test pipeline behavior
    processed_texts = pipeline.batch_embed(["text1", "text2", "text3"])
    assert len(processed_texts) == 3
    assert all(len(emb) == 1024 for emb in processed_texts)
```

#### Query Engine Boundary

```python
@pytest.fixture
def query_engine_boundary(ai_stack_boundary, temp_settings):
    """
    Boundary testing for query engines and retrieval systems.
    """
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.vector_stores import SimpleVectorStore
    
    # Create simple in-memory index for testing
    vector_store = SimpleVectorStore()
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=ai_stack_boundary['embed_model']
    )
    
    # Create query engine
    query_engine = index.as_query_engine(
        llm=ai_stack_boundary['llm'],
        similarity_top_k=3
    )
    
    yield {
        'index': index,
        'query_engine': query_engine,
        'vector_store': vector_store,
        'similarity_top_k': 3,
        'expected_response_type': str,
    }
```

### 3. HTTP Client Boundary Pattern

**Pattern**: Use specialized libraries for HTTP mocking instead of manual patches.

#### Responses Library Pattern

```python
@pytest.fixture
def mock_http_client():
    """
    Modern HTTP mocking using responses library.
    Replaces: @patch('requests.get'), @patch('httpx.AsyncClient'), etc.
    """
    import responses
    
    with responses.RequestsMock() as rsps:
        # Common response patterns
        rsps.add(
            responses.GET,
            "http://api.test.com/health",
            json={"status": "healthy", "version": "1.0.0"},
            status=200
        )
        
        rsps.add(
            responses.POST,
            "http://api.test.com/process",
            json={"result": "processed", "id": "test-123"},
            status=201
        )
        
        rsps.add(
            responses.GET,
            "http://api.test.com/models/available",
            json={"models": ["gpt-4", "claude-3", "llama-2"]},
            status=200
        )
        
        yield rsps
```

**Usage Example**:

```python
@pytest.mark.integration
def test_external_api_integration(mock_http_client):
    """Test external API integration with responses library."""
    client = ExternalAPIClient(base_url="http://api.test.com")
    
    # Test health check
    health = client.check_health()
    assert health["status"] == "healthy"
    
    # Test document processing
    result = client.process_document("test document content")
    assert result["result"] == "processed"
    assert result["id"] == "test-123"
    
    # Test model availability
    models = client.get_available_models()
    assert "gpt-4" in models["models"]
    assert len(models["models"]) == 3
```

#### Async HTTP Pattern (pytest-httpx)

```python
@pytest.fixture
async def async_http_boundary():
    """
    Async HTTP testing with pytest-httpx.
    For testing async HTTP operations.
    """
    # This would typically be handled by pytest-httpx plugin
    # but shown here for pattern illustration
    mock_responses = {
        "GET /api/data": {"data": [1, 2, 3], "status": "ok"},
        "POST /api/submit": {"id": "12345", "status": "accepted"},
        "GET /api/stream": {"stream": True, "chunks": 5}
    }
    
    yield mock_responses
```

### 4. Database Boundary Pattern

**Pattern**: Use real database connections with temporary data for integration tests.

#### Temporary Database Pattern

```python
@pytest.fixture
def temp_database(tmp_path):
    """
    Temporary SQLite database for integration testing.
    Provides real database behavior without external dependencies.
    """
    db_path = tmp_path / "test.db"
    database_url = f"sqlite:///{db_path}"
    
    # Create database with schema
    from sqlalchemy import create_engine, MetaData
    engine = create_engine(database_url)
    metadata = MetaData()
    
    # Create tables (simplified)
    from sqlalchemy import Table, Column, Integer, String, Text, DateTime
    documents_table = Table(
        'documents', metadata,
        Column('id', Integer, primary_key=True),
        Column('title', String(200)),
        Column('content', Text),
        Column('created_at', DateTime),
    )
    
    chunks_table = Table(
        'chunks', metadata,
        Column('id', Integer, primary_key=True), 
        Column('document_id', Integer),
        Column('content', Text),
        Column('embedding', Text),  # JSON serialized
    )
    
    metadata.create_all(engine)
    
    yield {
        'engine': engine,
        'database_url': database_url,
        'documents_table': documents_table,
        'chunks_table': chunks_table,
    }
    
    # Cleanup happens automatically with tmp_path
```

**Usage Example**:

```python
@pytest.mark.integration
def test_document_storage_workflow(temp_database):
    """Test document storage with real database operations."""
    storage = DocumentStorage(database_url=temp_database['database_url'])
    
    # Test document creation
    doc_id = storage.create_document(
        title="Test Document",
        content="This is test content for document storage."
    )
    
    assert doc_id is not None
    
    # Test document retrieval
    document = storage.get_document(doc_id)
    assert document['title'] == "Test Document"
    assert document['content'] == "This is test content for document storage."
    
    # Test document search
    results = storage.search_documents(query="test content")
    assert len(results) == 1
    assert results[0]['id'] == doc_id
```

### 5. File System Boundary Pattern

**Pattern**: Use temporary directories and files for filesystem testing.

#### Temporary Directory Pattern

```python
@pytest.fixture
def temp_filesystem(tmp_path):
    """
    Temporary filesystem setup for file operations testing.
    Prevents mock directory creation bugs and provides realistic file operations.
    """
    # Create directory structure
    cache_dir = tmp_path / "cache"
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    temp_dir = tmp_path / "temp"
    
    # Create directories
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample files
    sample_doc = data_dir / "sample.txt"
    sample_doc.write_text("Sample document content for testing.")
    
    config_file = cache_dir / "config.json"
    config_file.write_text('{"setting": "test_value"}')
    
    yield {
        'root': tmp_path,
        'cache_dir': str(cache_dir),
        'data_dir': str(data_dir),
        'logs_dir': str(logs_dir),
        'temp_dir': str(temp_dir),
        'sample_document': str(sample_doc),
        'config_file': str(config_file),
    }
    
    # Cleanup happens automatically with tmp_path
```

**Usage Example**:

```python
@pytest.mark.unit
def test_file_processing_operations(temp_filesystem):
    """Test file processing with real filesystem operations."""
    processor = FileProcessor(
        cache_dir=temp_filesystem['cache_dir'],
        data_dir=temp_filesystem['data_dir']
    )
    
    # Test file reading
    content = processor.read_file(temp_filesystem['sample_document'])
    assert content == "Sample document content for testing."
    
    # Test file processing and caching
    result = processor.process_and_cache("test_file.txt", content)
    assert result.success
    
    # Verify cache file was created
    cache_path = Path(temp_filesystem['cache_dir']) / "processed" / "test_file.txt"
    assert cache_path.exists()
    
    # Test file cleanup
    processor.cleanup_temp_files()
    temp_files = list(Path(temp_filesystem['temp_dir']).glob("*"))
    assert len(temp_files) == 0
```

## Library-First Testing Patterns

### 1. HTTP Testing with responses

**Replace**: Manual HTTP mocking with `@patch('requests.get')`

**With**: responses library for declarative HTTP mocking

```python
import responses

@pytest.mark.integration
def test_api_workflow_with_responses():
    """Test complete API workflow using responses library."""
    
    @responses.activate
    def run_test():
        # Setup API responses declaratively
        responses.add(
            responses.GET,
            "http://api.example.com/auth/token",
            json={"token": "test-token-123", "expires_in": 3600},
            status=200
        )
        
        responses.add(
            responses.POST,
            "http://api.example.com/documents/analyze",
            json={"analysis_id": "analysis-456", "status": "processing"},
            status=202
        )
        
        responses.add(
            responses.GET,
            "http://api.example.com/analysis/analysis-456/status",
            json={"status": "completed", "results": {"confidence": 0.95}},
            status=200
        )
        
        # Test business workflow
        client = DocumentAnalysisClient()
        
        # Authentication
        token = client.authenticate()
        assert token == "test-token-123"
        
        # Submit analysis
        analysis_id = client.analyze_document("test document content")
        assert analysis_id == "analysis-456"
        
        # Check results
        results = client.get_analysis_results(analysis_id)
        assert results["status"] == "completed"
        assert results["results"]["confidence"] == 0.95
    
    run_test()
```

### 2. Property-Based Testing with Hypothesis

**Pattern**: Use Hypothesis for edge case discovery and input validation testing.

```python
from hypothesis import given, strategies as st, assume

@pytest.mark.unit
@given(
    text=st.text(min_size=1, max_size=10000),
    chunk_size=st.integers(min_value=10, max_value=1000),
    overlap=st.integers(min_value=0, max_value=100)
)
def test_document_chunking_properties(temp_settings, text, chunk_size, overlap):
    """Property-based testing for document chunking logic."""
    assume(overlap < chunk_size)  # Valid constraint
    
    processor = DocumentProcessor(settings=temp_settings)
    chunks = processor.chunk_text(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    # Properties that must always hold
    if text.strip():  # Non-empty text
        assert len(chunks) > 0, "Should produce at least one chunk for non-empty text"
        
        # All chunks should be within size limit
        assert all(len(chunk) <= chunk_size for chunk in chunks), \
            "All chunks should be within size limit"
        
        # First and last chunks should contain original text boundaries
        assert chunks[0].startswith(text[:min(len(text), 20)]), \
            "First chunk should start with beginning of text"
        
        # Reconstruct text should be similar to original (accounting for overlap)
        if overlap == 0:
            reconstructed = ''.join(chunks)
            assert reconstructed == text, "Zero overlap should preserve full text"
```

### 3. Container Testing with testcontainers

**Pattern**: Use real service containers for integration testing.

```python
@pytest.fixture(scope="session")
def qdrant_container():
    """Real Qdrant container for integration testing."""
    from testcontainers.compose import DockerCompose
    
    with DockerCompose("./", compose_file_name="docker-compose.test.yml") as compose:
        qdrant_port = compose.get_service_port("qdrant", 6333)
        qdrant_url = f"http://localhost:{qdrant_port}"
        
        # Wait for service readiness
        import requests
        import time
        for _ in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f"{qdrant_url}/health")
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            pytest.fail("Qdrant container failed to start")
        
        yield {
            'url': qdrant_url,
            'port': qdrant_port,
            'health_endpoint': f"{qdrant_url}/health"
        }
```

### 4. Performance Testing with pytest-benchmark

**Pattern**: Integrated performance benchmarking with statistical analysis.

```python
@pytest.mark.performance
def test_embedding_generation_performance(benchmark, ai_stack_boundary):
    """Benchmark embedding generation with statistical analysis."""
    embed_model = ai_stack_boundary['embed_model']
    
    def generate_embeddings():
        """Function to benchmark."""
        texts = [
            "Artificial intelligence research document",
            "Machine learning algorithm implementation", 
            "Deep learning neural network architecture",
            "Natural language processing pipeline",
            "Computer vision model training"
        ]
        
        return [embed_model.get_text_embedding(text) for text in texts]
    
    # Run benchmark with statistical analysis
    result = benchmark.pedantic(
        generate_embeddings,
        iterations=10,    # Number of iterations
        rounds=5,         # Number of rounds
        warmup_rounds=2   # Warmup iterations
    )
    
    # Performance assertions
    assert len(result) == 5
    assert all(len(emb) == 1024 for emb in result)
    
    # Access benchmark stats
    stats = benchmark.stats
    assert stats.mean < 0.1, f"Mean execution time {stats.mean}s exceeds threshold"
    assert stats.stddev < 0.02, f"Standard deviation {stats.stddev} too high"
```

## Error Testing Patterns

### 1. Exception Boundary Testing

**Pattern**: Test error conditions at system boundaries with realistic scenarios.

```python
@pytest.mark.unit
def test_resource_exhaustion_handling(system_resource_boundary, logging_boundary):
    """Test resource exhaustion error handling."""
    
    # Simulate low memory condition
    system_resource_boundary['virtual_memory'].available = 100 * 1024 * 1024  # 100MB
    system_resource_boundary['virtual_memory'].percent = 95.0
    
    processor = DocumentProcessor()
    
    # Test resource exhaustion detection
    with pytest.raises(ResourceExhaustionError, match="Insufficient memory"):
        processor.process_large_document("x" * (500 * 1024 * 1024))  # 500MB doc
    
    # Verify error logging
    logging_boundary['assert_error_called']("Resource exhaustion detected")
    
    # Test graceful degradation
    result = processor.process_with_fallback("x" * (500 * 1024 * 1024))
    assert result.status == "completed"
    assert result.method == "fallback"
    assert "limited memory" in result.warnings
```

### 2. Network Error Simulation

```python
@pytest.mark.integration
def test_network_error_recovery(mock_http_client):
    """Test network error recovery strategies."""
    
    # Setup failing then succeeding responses
    mock_http_client.add(
        responses.GET,
        "http://external-service.com/api/process",
        json={"error": "Service temporarily unavailable"},
        status=503
    )
    
    mock_http_client.add(
        responses.GET, 
        "http://external-service.com/api/process",
        json={"result": "processed successfully", "id": "retry-123"},
        status=200
    )
    
    client = ResilientAPIClient(
        base_url="http://external-service.com",
        retry_count=3,
        retry_delay=0.1  # Fast retry for testing
    )
    
    # Test retry behavior
    result = client.process_with_retry("test data")
    
    # Should succeed after retry
    assert result["result"] == "processed successfully"
    assert result["id"] == "retry-123"
    
    # Verify retry happened
    assert len(mock_http_client.calls) == 2
    assert mock_http_client.calls[0].response.status_code == 503
    assert mock_http_client.calls[1].response.status_code == 200
```

### 3. Data Corruption Testing

```python
@pytest.mark.unit
def test_data_corruption_detection(temp_filesystem):
    """Test data corruption detection and recovery."""
    
    # Create corrupted file
    corrupted_file = Path(temp_filesystem['data_dir']) / "corrupted.json"
    corrupted_file.write_text('{"incomplete": "json data"...')  # Invalid JSON
    
    processor = DataProcessor(data_dir=temp_filesystem['data_dir'])
    
    # Test corruption detection
    with pytest.raises(DataCorruptionError, match="Invalid JSON format"):
        processor.load_configuration(str(corrupted_file))
    
    # Test recovery with backup
    backup_file = Path(temp_filesystem['data_dir']) / "corrupted.json.backup"
    backup_file.write_text('{"complete": "valid json data"}')
    
    result = processor.load_configuration_with_backup(str(corrupted_file))
    assert result["complete"] == "valid json data"
    assert processor.last_recovery_action == "backup_restore"
```

## Test Data Management Patterns

### 1. Test Data Registry Pattern

```python
@pytest.fixture(scope="session")
def test_data_registry():
    """Centralized test data for consistent testing across modules."""
    return {
        "documents": {
            "minimal": {
                "content": "AI systems require careful design.",
                "metadata": {"title": "AI Design", "words": 6}
            },
            "standard": {
                "content": "Machine learning pipelines process data efficiently through multiple stages including data collection, preprocessing, feature extraction, model training, and evaluation.",
                "metadata": {"title": "ML Pipelines", "words": 22}
            },
            "complex": {
                "content": """
                Deep learning neural networks have revolutionized artificial intelligence by enabling computers to learn complex patterns from large datasets. These networks consist of multiple layers of interconnected nodes that process information hierarchically. The training process involves backpropagation algorithms that adjust connection weights to minimize prediction errors. Applications span computer vision, natural language processing, speech recognition, and autonomous systems.
                """.strip(),
                "metadata": {"title": "Deep Learning Overview", "words": 56}
            }
        },
        "queries": {
            "factual": "What is artificial intelligence?",
            "analytical": "Compare the effectiveness of different machine learning approaches",
            "summarization": "Summarize the key points about neural network training",
            "complex": "Analyze the document and provide recommendations for improving the described system"
        },
        "embeddings": {
            "ai_concept": [0.1] * 1024,      # Embedding for AI-related content
            "ml_concept": [0.2] * 1024,      # Embedding for ML-related content  
            "technical": [0.3] * 1024,       # Embedding for technical content
            "general": [0.05] * 1024         # Embedding for general content
        },
        "expected_results": {
            "chunk_counts": {"minimal": 1, "standard": 1, "complex": 3},
            "processing_times": {"factual": 1.0, "analytical": 3.0, "complex": 5.0},
            "confidence_thresholds": {"high": 0.8, "medium": 0.6, "low": 0.4}
        }
    }
```

### 2. Dynamic Test Data Generation

```python
from hypothesis import strategies as st

@pytest.fixture
def document_generator():
    """Generate realistic test documents with controlled properties."""
    
    def generate_document(
        word_count: int = 100,
        complexity: str = "medium", 
        topic: str = "technology",
        include_metadata: bool = True
    ):
        """Generate document with specified characteristics."""
        
        # Word pools by topic
        word_pools = {
            "technology": ["artificial", "intelligence", "machine", "learning", "algorithm", "neural", "network", "data", "processing", "analysis"],
            "science": ["research", "hypothesis", "experiment", "methodology", "results", "conclusion", "analysis", "evidence", "theory", "validation"],
            "business": ["strategy", "market", "customer", "revenue", "growth", "innovation", "competitive", "analysis", "optimization", "performance"]
        }
        
        # Complexity patterns
        complexity_patterns = {
            "simple": {"avg_sentence_length": 8, "technical_terms_ratio": 0.1},
            "medium": {"avg_sentence_length": 15, "technical_terms_ratio": 0.3}, 
            "complex": {"avg_sentence_length": 25, "technical_terms_ratio": 0.5}
        }
        
        words = word_pools.get(topic, word_pools["technology"])
        pattern = complexity_patterns.get(complexity, complexity_patterns["medium"])
        
        # Generate content
        content = []
        current_words = 0
        
        while current_words < word_count:
            sentence_length = min(
                pattern["avg_sentence_length"],
                word_count - current_words
            )
            
            sentence_words = []
            for _ in range(sentence_length):
                if random.random() < pattern["technical_terms_ratio"]:
                    sentence_words.append(random.choice(words))
                else:
                    sentence_words.append(random.choice(["the", "and", "of", "to", "in", "for", "with", "by", "from", "on"]))
            
            content.append(" ".join(sentence_words).capitalize() + ".")
            current_words += sentence_length
        
        document = {
            "content": " ".join(content),
            "word_count": current_words,
            "complexity": complexity,
            "topic": topic
        }
        
        if include_metadata:
            document["metadata"] = {
                "generated": True,
                "word_count": current_words,
                "complexity_level": complexity,
                "topic_category": topic,
                "estimated_processing_time": current_words * 0.01  # seconds
            }
        
        return document
    
    return generate_document
```

## Performance Testing Patterns

### 1. Regression Detection Pattern

```python
@pytest.fixture
def performance_baseline():
    """Load performance baselines for regression detection."""
    import json
    
    baseline_file = Path("tests/performance/baselines.json")
    if baseline_file.exists():
        baselines = json.loads(baseline_file.read_text())
    else:
        # Default baselines if file doesn't exist
        baselines = {
            "embedding_generation": {"mean": 0.05, "max": 0.1},
            "document_processing": {"mean": 0.5, "max": 1.0},
            "query_execution": {"mean": 0.2, "max": 0.5},
            "index_creation": {"mean": 2.0, "max": 5.0}
        }
    
    return baselines

@pytest.mark.performance
def test_embedding_performance_regression(benchmark, ai_stack_boundary, performance_baseline):
    """Test embedding performance against established baseline."""
    embed_model = ai_stack_boundary['embed_model']
    baseline = performance_baseline["embedding_generation"]
    
    def embed_operation():
        return embed_model.get_text_embedding("Test document for performance measurement")
    
    # Run benchmark
    result = benchmark(embed_operation)
    
    # Regression detection
    mean_time = benchmark.stats.mean
    max_time = benchmark.stats.max
    
    assert mean_time <= baseline["mean"] * 1.1, \
        f"Mean time {mean_time}s exceeds baseline {baseline['mean']}s by >10%"
    
    assert max_time <= baseline["max"] * 1.2, \
        f"Max time {max_time}s exceeds baseline {baseline['max']}s by >20%"
    
    # Update baseline if significantly better
    if mean_time < baseline["mean"] * 0.9:
        logger.info(f"Performance improved: {mean_time}s vs baseline {baseline['mean']}s")
```

### 2. Memory Usage Testing Pattern

```python
@pytest.mark.performance
def test_memory_usage_bounds(temp_settings):
    """Test memory usage stays within acceptable bounds during processing."""
    import psutil
    import gc
    
    # Get baseline memory usage
    process = psutil.Process()
    baseline_memory = process.memory_info().rss
    
    processor = DocumentProcessor(settings=temp_settings)
    
    # Process multiple documents and measure memory growth
    memory_measurements = []
    
    for i in range(50):  # Process 50 documents
        document_content = f"Test document {i} content " * 100  # ~2KB each
        processor.process_document(document_content)
        
        # Measure memory every 10 documents
        if i % 10 == 0:
            gc.collect()  # Force garbage collection
            current_memory = process.memory_info().rss
            memory_growth = current_memory - baseline_memory
            memory_measurements.append({
                'document_count': i + 1,
                'memory_growth_mb': memory_growth / (1024 * 1024),
                'memory_per_doc_kb': memory_growth / (1024 * (i + 1))
            })
    
    # Assert memory usage constraints
    final_measurement = memory_measurements[-1]
    
    # Total memory growth should be reasonable
    assert final_measurement['memory_growth_mb'] < 100, \
        f"Memory growth {final_measurement['memory_growth_mb']:.2f}MB exceeds 100MB limit"
    
    # Memory per document should be bounded
    assert final_measurement['memory_per_doc_kb'] < 500, \
        f"Memory per document {final_measurement['memory_per_doc_kb']:.2f}KB exceeds 500KB limit"
    
    # Memory growth should be roughly linear (no significant leaks)
    growth_rates = [m['memory_per_doc_kb'] for m in memory_measurements]
    growth_variance = max(growth_rates) - min(growth_rates)
    
    assert growth_variance < 100, \
        f"Memory growth variance {growth_variance:.2f}KB suggests memory leak"
```

## Test Organization Patterns

### 1. Hierarchical Test Structure

```
tests/
├── unit/                           # Tier 1: Fast, isolated tests
│   ├── test_core/                  # Core business logic
│   │   ├── test_document_processing.py
│   │   ├── test_embedding_logic.py
│   │   └── test_query_processing.py
│   ├── test_agents/                # Agent system components
│   │   ├── test_coordinator.py
│   │   ├── test_planning_agent.py
│   │   └── test_retrieval_agent.py
│   └── test_utils/                 # Utility functions
├── integration/                    # Tier 2: Component interaction
│   ├── test_pipelines/             # End-to-end pipelines
│   ├── test_agent_coordination/    # Multi-agent workflows
│   └── test_external_services/     # External service integration
├── system/                         # Tier 3: Full system tests
│   ├── test_gpu_workflows.py       # GPU-dependent tests
│   └── test_production_scenarios.py
├── performance/                    # Performance benchmarks
│   ├── test_latency_benchmarks.py
│   ├── test_memory_benchmarks.py
│   └── test_throughput_benchmarks.py
└── validation/                     # Production readiness
    ├── test_configuration.py
    └── test_environment_setup.py
```

### 2. Marker-Based Organization

```python
# Mark tests by functionality and requirements
@pytest.mark.unit
@pytest.mark.agents
@pytest.mark.requires_gpu
def test_gpu_agent_coordination():
    """GPU-dependent unit test for agent coordination."""
    pass

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.performance
def test_full_pipeline_performance():
    """Integration test with performance requirements."""
    pass

# Mark tests by stability
@pytest.mark.stable      # Tests that consistently pass
@pytest.mark.flaky       # Tests that sometimes fail
@pytest.mark.experimental  # New tests being validated
```

### 3. Parameterized Test Patterns

```python
@pytest.mark.parametrize("document_type,expected_chunks", [
    ("simple", 1),
    ("medium", 3), 
    ("complex", 8),
    ("very_large", 20)
])
@pytest.mark.unit
def test_chunking_by_document_type(temp_settings, test_data_registry, document_type, expected_chunks):
    """Test document chunking across different document types."""
    document = test_data_registry["documents"][document_type]
    processor = DocumentProcessor(settings=temp_settings)
    
    chunks = processor.chunk_document(
        content=document["content"],
        chunk_size=100,
        overlap=20
    )
    
    assert len(chunks) == expected_chunks
    assert all(len(chunk.content) <= 100 for chunk in chunks)

@pytest.mark.parametrize("query_complexity,timeout", [
    ("simple", 1.0),
    ("moderate", 5.0),
    ("complex", 15.0),
])  
@pytest.mark.integration
def test_query_processing_timeouts(ai_stack_boundary, query_complexity, timeout):
    """Test query processing respects complexity-based timeouts."""
    query_engine = QueryEngine(
        llm=ai_stack_boundary['llm'],
        embed_model=ai_stack_boundary['embed_model']
    )
    
    test_queries = {
        "simple": "What is AI?",
        "moderate": "Compare machine learning approaches for document analysis",
        "complex": "Analyze the document structure, extract key insights, and provide detailed recommendations with supporting evidence"
    }
    
    start_time = time.time()
    result = query_engine.process_query(
        query=test_queries[query_complexity],
        timeout=timeout
    )
    processing_time = time.time() - start_time
    
    assert result is not None
    assert processing_time <= timeout
```

These testing patterns provide a comprehensive foundation for maintainable, reliable testing in DocMind AI. The emphasis on boundary testing, library-first approaches, and realistic data ensures tests remain valuable and stable as the system evolves.