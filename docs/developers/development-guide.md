# Development Guide

## Overview

This comprehensive guide covers development practices, coding standards, framework usage, and testing procedures for DocMind AI. Follow these guidelines to ensure consistent, high-quality contributions to the project.

> **Prerequisites**: Complete [Developer Setup](developer-setup.md) before following this guide.

## Getting Started

### Prerequisites

Ensure you have completed the [Developer Setup](developer-setup.md) guide before proceeding. This includes:

- Python 3.10+ environment setup
- Required dependencies installed via `uv sync`
- Services running (Qdrant, etc.)
- Configuration validated

### Development Environment

With setup complete, you can start development:

```bash
# Verify setup
python -c "from src.config import settings; print(f'✅ {settings.app_name} ready')"

# Start development server
streamlit run src/app.py
```

### Project Structure

```text
docmind-ai/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── utils/              # Core utilities and processing
│   └── agents/             # Multi-agent coordination system
├── tests/                  # Test suite
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
└── .env.example            # Environment template
```

## Coding Standards

### Python Code Style

- **Style:** Use Google-style docstrings, type hints, and max line length of 88
- **Linting:** Run `ruff check .` and `ruff format .` (configured in `pyproject.toml`)
  - Selected rules: E, F, I, UP, N, S, B, A, C4, PT, SIM, TID, D
  - Ignored: D203, D213, S301, S603, S607, S108

### Core Principles

- **KISS (Keep It Simple, Stupid):** Prefer simple, readable solutions over complex ones
- **DRY (Don't Repeat Yourself):** Eliminate code duplication through functions and modules
- **YAGNI (You Aren't Gonna Need It):** Don't implement features until they're actually needed
- **Library-First:** Prefer well-established libraries over custom implementations

### Code Quality Guidelines

```python
# Good: Clear function with type hints and docstring
def process_documents(files: List[Path], chunk_size: int = 1000) -> List[Document]:
    """Process uploaded documents into chunks for analysis.
    
    Args:
        files: List of file paths to process
        chunk_size: Size of text chunks for processing
        
    Returns:
        List of processed document chunks
    """
    documents = []
    for file in files:
        content = load_file_content(file)
        chunks = split_content(content, chunk_size)
        documents.extend(chunks)
    return documents

# Bad: No types, unclear purpose, too complex
def proc(f, s=1000):
    d = []
    for x in f:
        c = open(x).read()
        if len(c) > s:
            for i in range(0, len(c), s):
                d.append(c[i:i+s])
        else:
            d.append(c)
    return d
```

### Dependency Management

- **Dependencies:** Use `uv` for package management
- **Library Selection:** Prefer mature, well-documented libraries
- **Version Pinning:** Pin major versions for stability
- **Extras:** Group optional dependencies appropriately

```toml
# pyproject.toml example
[project.optional-dependencies]
test = ["pytest>=8.0", "pytest-cov"]
gpu = ["torch>=2.7.1", "vllm>=0.10.1"]
```

## LangChain Framework Usage

### Overview - LangChain

LangChain is the core orchestration framework in DocMind AI, handling LLM interactions, document processing, retrieval, and chaining.

- **Version:** langchain>=0.3.26, langchain-community==0.3.27
- **Purpose:** Modular components for chains, retrievers, splitters, embeddings, and vector stores
- **Integration Points:** Primarily in `src/utils/` for processing and `src/app.py` for invoking analysis/chat

### Key Components and Usage

#### 1. Chains

**LLMChain:** Used in `analyze_documents()` for prompt-based analysis with custom tones/instructions.

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Create a chain for document analysis
prompt = PromptTemplate(
    input_variables=["text", "tone"],
    template="Analyze this document with a {tone} tone: {text}"
)
chain = LLMChain(llm=llm, prompt=prompt)
output = chain.run(text=document_text, tone="professional")
```

**RetrievalQA:** In `chat_with_context()` for RAG-based chat responses.

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=compression_retriever
)
response = qa_chain.run(query="What are the key findings?")
```

**load_summarize_chain:** For map-reduce summarization of large texts.

```python
from langchain.chains.summarize import load_summarize_chain

sum_chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = sum_chain.run(document_chunks)
```

#### 2. Retrievers

**ContextualCompressionRetriever:** Wraps base retrievers with custom compressors for reranking.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import JinaRerankCompressor

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(
        search_type="hybrid", 
        search_kwargs={"k": 10}
    )
)
```

#### 3. Text Splitters

**RecursiveCharacterTextSplitter:** Splits documents for processing and chunking.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
```

#### 4. Embeddings

**HuggingFaceEmbeddings:** For dense embeddings in vectorstore creation.

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

dense_embeddings = HuggingFaceEmbeddings(
    model_name=settings.default_embedding_model,
    model_kwargs={"device_map": "auto"}
)
```

**FastEmbedSparse:** For sparse embeddings in hybrid search.

```python
from langchain_community.embeddings import FastEmbedSparse

sparse_embeddings = FastEmbedSparse(
    model_name="prithivida/Splade_PP_en_v1",
    providers=["CUDAExecutionProvider"] if device == 'cuda' else None
)
```

#### 5. Vector Stores

**QdrantVectorStore:** Creates hybrid vectorstores for retrieval.

```python
from langchain_community.vectorstores import Qdrant

vectorstore = Qdrant.from_documents(
    documents=docs,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    client=client,
    collection_name="docmind",
    hybrid=True
)
```

#### 6. Output Parsers

**PydanticOutputParser:** Structures analysis outputs.

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class AnalysisOutput(BaseModel):
    summary: str
    key_points: List[str]
    conclusions: str

parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
parsed_output = parser.parse(llm_response)
```

### LangChain Best Practices

#### RAG Pipeline Design

```python
def create_rag_pipeline(documents: List[Document], llm) -> RetrievalQA:
    """Create an optimized RAG pipeline with hybrid search."""
    
    # 1. Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # 2. Create embeddings
    dense_embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v3")
    sparse_embeddings = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")
    
    # 3. Create vectorstore
    vectorstore = Qdrant.from_documents(
        chunks, dense_embeddings, sparse_embedding=sparse_embeddings
    )
    
    # 4. Create retriever with compression
    base_retriever = vectorstore.as_retriever(search_type="hybrid", search_kwargs={"k": 10})
    compressor = JinaRerankCompressor(top_n=5)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # 5. Create QA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
```

#### Error Handling

```python
def safe_parse_output(parser: PydanticOutputParser, raw_output: str):
    """Safely parse LLM output with fallback."""
    try:
        return parser.parse(raw_output)
    except Exception as e:
        logger.warning(f"Parsing failed: {e}. Using raw output.")
        return {"raw_output": raw_output, "error": str(e)}
```

#### Extensibility Patterns

- **Modular Design:** Use LangChain's component interfaces for easy swapping
- **Configuration-Driven:** Make embeddings, retrievers, and chains configurable
- **Chain Composition:** Combine multiple chains for complex workflows

## Testing Framework

### Testing Overview

- **Tool:** pytest v8.3.1 (installed via `uv sync --extra test`)
- **Location:** Tests in `tests/` directory
- **Coverage:** Aim for >80% coverage on `src/utils/` and `src/app.py`

### Test Setup

1. **Install test dependencies:**

   ```bash
   uv sync --extra test
   ```

2. **Run tests:**

   ```bash
   pytest
   ```

3. **Run with coverage:**

   ```bash
   pytest --cov=docmind_ai --cov-report=html
   ```

### Writing Tests

#### Unit Tests

Target functions in `src/utils/` with clear inputs/outputs:

```python
import pytest
from pathlib import Path
from langchain.schema import Document
from src.utils.document_loader import load_documents

def test_load_pdf_documents(tmp_path):
    """Test loading PDF documents."""
    # Setup
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "w") as f:
        f.write("%PDF-1.4 dummy content")
    
    # Execute
    docs = load_documents([pdf_path])
    
    # Assert
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    assert docs[0].page_content is not None

def test_load_empty_file(tmp_path):
    """Test handling of empty files."""
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    
    docs = load_documents([empty_file])
    assert len(docs) == 0

def test_load_unsupported_format(tmp_path):
    """Test handling of unsupported file formats."""
    unsupported_file = tmp_path / "test.xyz"
    unsupported_file.write_text("content")
    
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_documents([unsupported_file])
```

#### Integration Tests

Test end-to-end workflows:

```python
from unittest.mock import Mock, patch
from src.app import analyze_documents
from src.models import AnalysisOutput

def test_analysis_pipeline():
    """Test complete document analysis pipeline."""
    # Setup mocks
    mock_llm = Mock()
    mock_llm.invoke.return_value = "Test analysis output"
    
    with patch("src.utils.load_documents") as mock_load:
        mock_load.return_value = [Document(page_content="test content")]
        
        # Execute
        result = analyze_documents(
            llm=mock_llm,
            file_paths=["test.pdf"],
            analysis_type="Comprehensive",
            custom_instruction="",
            tone="Neutral",
            persona="General Assistant",
            length="Concise",
            max_tokens=4096
        )
        
        # Assert
        assert isinstance(result, AnalysisOutput)
        assert result.summary is not None
```

#### Edge Cases and Error Handling

```python
def test_large_document_handling():
    """Test handling of large documents."""
    large_content = "x" * 10000000  # 10MB of text
    doc = Document(page_content=large_content)
    
    # Should not raise memory errors
    chunks = split_document(doc, chunk_size=1000)
    assert len(chunks) > 0

def test_gpu_fallback():
    """Test CPU fallback when GPU unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        embeddings = unified embedding configuration
        assert embeddings.device == "cpu"

def test_model_unavailable():
    """Test handling when Ollama model is unavailable."""
    with patch("requests.get") as mock_get:
        mock_get.side_effect = ConnectionError("Model unavailable")
        
        with pytest.raises(ConnectionError):
            initialize_llm("nonexistent-model")
```

### Test Best Practices

#### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

# Mock Ollama API calls
@patch("requests.post")
def test_llm_generation(mock_post):
    mock_post.return_value.json.return_value = {"response": "test output"}
    result = generate_response("test prompt")
    assert result == "test output"

# Mock file system operations
@patch("pathlib.Path.exists")
def test_file_validation(mock_exists):
    mock_exists.return_value = True
    assert validate_file_path("/fake/path") is True
```

#### Fixtures for Reusable Test Data

```python
@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        Document(page_content="Sample document 1"),
        Document(page_content="Sample document 2"),
    ]

@pytest.fixture
def mock_vectorstore():
    """Provide a mock vectorstore for testing."""
    mock_store = Mock()
    mock_store.similarity_search.return_value = [
        Document(page_content="relevant content")
    ]
    return mock_store

def test_retrieval_with_fixtures(sample_documents, mock_vectorstore):
    """Test retrieval using fixtures."""
    results = mock_vectorstore.similarity_search("query")
    assert len(results) > 0
```

#### Performance and Load Testing

```python
import time
import pytest

@pytest.mark.performance
def test_document_processing_speed():
    """Test document processing performance."""
    start_time = time.time()
    
    # Process large document set
    results = process_large_document_set()
    
    processing_time = time.time() - start_time
    assert processing_time < 30  # Should complete within 30 seconds
    assert len(results) > 0

@pytest.mark.slow
def test_concurrent_requests():
    """Test handling multiple concurrent requests."""
    import concurrent.futures
    
    def make_request():
        return analyze_document("test content")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [f.result() for f in futures]
    
    assert len(results) == 10
    assert all(r is not None for r in results)
```

## Development Workflow

### Before You Start Coding

1. **Read the relevant ADR** - Check `docs/adrs/` for architectural decisions
2. **Understand the configuration** - Always use `from src.config import settings`
3. **Check existing patterns** - Look for similar implementations first
4. **Plan your changes** - Keep it simple (KISS principle)

### Development Cycle

```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name

# 2. Make changes following patterns
# - Use settings for configuration
# - Follow import patterns
# - Keep functions simple (<20 lines ideal)

# 3. Test your changes
uv run python -m pytest tests/unit/test_your_feature.py -v

# 4. Quality checks
ruff format src tests
ruff check src tests --fix
ruff check src tests  # Must show "All checks passed!"

# 5. Integration testing
uv run python -m pytest tests/integration/ -k your_feature

# 6. Commit and push
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### Code Style Requirements

**Mandatory standards:**

```python
# 1. Always use type hints
def process_documents(file_paths: list[str]) -> list[Document]:
    """Process documents with proper typing."""
    return documents

# 2. Use Pydantic models for data
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(default=10, ge=1, le=50)

# 3. Configuration access pattern  
from src.config import settings

def get_model_name() -> str:
    return settings.vllm.model  # ✅ Correct

# 4. Async when appropriate
async def process_query(query: str) -> QueryResponse:
    """Process query asynchronously."""
    # Use async for I/O operations
    
# 5. Error handling
try:
    result = process_documents(files)
except DocumentProcessingError as e:
    logger.error("Document processing failed: %s", e)
    raise
```

### Common Development Tasks

#### Adding a New Feature

Example: Adding a new document format support

```python
# 1. Add configuration if needed
# In src/config/settings.py
class ProcessingConfig(BaseModel):
    # Existing fields...
    support_new_format: bool = Field(default=False)

# 2. Create the feature
# In src/processing/new_format_processor.py
from src.config import settings
from src.models.processing import ProcessingResult

class NewFormatProcessor:
    def __init__(self):
        self.enabled = settings.processing.support_new_format
    
    def process(self, file_path: str) -> ProcessingResult:
        if not self.enabled:
            raise ValueError("New format support disabled")
        # Implementation here...

# 3. Integrate with existing pipeline
# In src/processing/document_processor.py
from src.processing.new_format_processor import NewFormatProcessor

class DocumentProcessor:
    def __init__(self):
        self.new_format_processor = NewFormatProcessor()
    
    def process(self, file_path: str) -> ProcessingResult:
        if file_path.endswith('.newformat'):
            return self.new_format_processor.process(file_path)
        # Existing logic...

# 4. Add tests
# In tests/unit/test_new_format.py
def test_new_format_processing():
    from src.processing.new_format_processor import NewFormatProcessor
    processor = NewFormatProcessor()
    # Test implementation...
```

## Contribution Guidelines

### Branch Management

1. **Create a Branch:**

   ```bash
   git checkout -b feat/<feature-name>  # or fix/<bug-name>
   ```

2. **Branch Naming Convention:**
   - `feat/` - New features
   - `fix/` - Bug fixes  
   - `docs/` - Documentation updates
   - `test/` - Test additions/improvements
   - `refactor/` - Code refactoring

### Development Process

1. **Develop and Test:**
   - Write code in appropriate modules (`src/app.py`, `src/utils/`, etc.)
   - Add comprehensive tests in `tests/`
   - Ensure all tests pass: `pytest`
   - Run linting: `ruff check . && ruff format .`

2. **Commit Standards:**
   - Use clear, descriptive commit messages
   - Follow conventional commits when possible
   - Example: "feat: add hybrid search with Jina v4 and SPLADE++"

3. **Code Quality Checks:**

   ```bash
   # Run all quality checks before committing
   ruff check .           # Linting
   ruff format .          # Formatting
   pytest                 # Tests
   pytest --cov=src       # Coverage check
   ```

### Pull Request Process

1. **Push and Create PR:**

   ```bash
   git push origin feat/<feature-name>
   ```

2. **PR Requirements:**
   - Descriptive title and detailed description
   - Link to related issues
   - All CI checks must pass
   - Code review approval required

3. **Code Review Guidelines:**
   - Address feedback promptly and thoroughly
   - Explain design decisions when requested
   - Update tests and documentation as needed
   - Ensure backward compatibility when possible

## Deployment

### Local Development

```bash
# Standard local development
streamlit run src/app.py

# With specific environment
ENV=development streamlit run src/app.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Build specific service
docker-compose build app

# Run in detached mode
docker-compose up -d
```

### Environment Configuration

Ensure environment variables are properly set (see `.env.example`):

```bash
# Copy and customize environment template
cp .env.example .env

# Required variables
OLLAMA_BASE_URL=http://localhost:11434
QDRANT_URL=http://localhost:6333
LOG_LEVEL=INFO
```

## Error Handling and Debugging

### Logging Standards

```python
import structlog

logger = structlog.get_logger(__name__)

def process_document(file_path: Path) -> Document:
    """Process a document with proper logging."""
    logger.info("Starting document processing", file_path=str(file_path))
    
    try:
        content = load_file_content(file_path)
        logger.debug("File loaded successfully", size=len(content))
        
        document = parse_content(content)
        logger.info("Document processed successfully", pages=document.page_count)
        
        return document
        
    except Exception as e:
        logger.error("Document processing failed", error=str(e), file_path=str(file_path))
        raise
```

### Error Recovery Patterns

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def robust_api_call(data):
    """API call with automatic retry logic."""
    try:
        return make_api_request(data)
    except TemporaryError as e:
        logger.warning("Temporary error, retrying", error=str(e))
        raise
    except PermanentError as e:
        logger.error("Permanent error, not retrying", error=str(e))
        raise
```

## Performance Optimization

### Code Performance

- **Async Operations:** Use async/await for I/O operations
- **Batch Processing:** Process documents in batches when possible
- **Memory Management:** Use generators for large datasets
- **Caching:** Implement result caching for expensive operations

### Example Optimizations

```python
import asyncio
from typing import AsyncGenerator

async def process_documents_async(files: List[Path]) -> AsyncGenerator[Document, None]:
    """Process documents asynchronously."""
    semaphore = asyncio.Semaphore(5)  # Limit concurrent operations
    
    async def process_single_file(file_path: Path) -> Document:
        async with semaphore:
            return await load_document_async(file_path)
    
    tasks = [process_single_file(file) for file in files]
    
    for coro in asyncio.as_completed(tasks):
        yield await coro

# Memory-efficient processing
def process_large_dataset(data_source):
    """Process large datasets without loading everything into memory."""
    for batch in batch_generator(data_source, batch_size=100):
        yield process_batch(batch)
```

## Documentation Standards

### Code Documentation

```python
def analyze_sentiment(text: str, model_name: str = "default") -> Dict[str, float]:
    """Analyze sentiment of input text using specified model.
    
    This function performs sentiment analysis on the provided text using
    the specified model. Returns confidence scores for different sentiment
    categories.
    
    Args:
        text: Input text to analyze for sentiment
        model_name: Name of the sentiment analysis model to use.
                   Defaults to "default" which uses the configured model.
    
    Returns:
        Dictionary containing sentiment scores:
        - positive: Confidence score for positive sentiment (0.0-1.0)
        - negative: Confidence score for negative sentiment (0.0-1.0)
        - neutral: Confidence score for neutral sentiment (0.0-1.0)
    
    Raises:
        ValueError: If text is empty or model_name is invalid
        ModelNotFoundError: If specified model is not available
    
    Example:
        >>> analyze_sentiment("I love this product!")
        {'positive': 0.92, 'negative': 0.03, 'neutral': 0.05}
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    # Implementation here
    pass
```

### README Updates

When adding new features, update relevant documentation:

- `README.md` - Overall project description and quick start
- `docs/developers/` - Detailed technical documentation
- `CHANGELOG.md` - Version history and changes
- Code comments - Inline explanations for complex logic

## Community and Support

### Reporting Issues

Use GitHub Issues for:

- Bug reports with detailed reproduction steps
- Feature requests with clear use cases
- Questions about implementation or usage
- Documentation improvements

### Issue Template

```markdown
## Bug Report / Feature Request

**Description:**
Clear description of the issue or requested feature

**Steps to Reproduce (for bugs):**
1. Step one
2. Step two
3. Expected vs actual behavior

**Environment:**
- OS: [e.g., Ubuntu 24.04]
- Python version: [e.g., 3.12]
- DocMind AI version: [e.g., v1.0.0]

**Logs:**
Include relevant logs from `logs/app.log`

**Additional Context:**
Any other relevant information
```

### Getting Help

- **GitHub Discussions:** For general questions and community support
- **Issues:** For specific bugs or feature requests
- **LinkedIn:** Contact maintainers directly via [LinkedIn](https://www.linkedin.com/in/bjorn-melin/)

## Continuous Integration

### GitHub Actions

The project uses GitHub Actions for:

- **Code Quality:** `ruff check` and `ruff format` validation
- **Testing:** `pytest` execution with coverage reporting
- **Security:** Dependency vulnerability scanning
- **Documentation:** Automated documentation building

### Pre-commit Checks

Ensure these pass before submitting PRs:

```bash
# Quality checks
ruff check .
ruff format .

# Test suite
pytest

# Coverage check
pytest --cov=src --cov-report=term-missing
```

## Summary

This development guide provides the foundation for contributing to DocMind AI. Follow these practices to ensure:

- **Code Quality:** Consistent, readable, and maintainable code
- **Framework Mastery:** Effective use of LangChain components
- **Testing Excellence:** Comprehensive test coverage and reliable validation
- **Collaboration:** Smooth contribution workflow and community engagement

For specific technical details, refer to other documentation files:

- [GPU and Performance](gpu-and-performance.md) - Hardware optimization
- [Multi-Agent System](multi-agent-system.md) - Agent coordination
- [Architecture](architecture.md) - System design overview

Thank you for contributing to DocMind AI!
