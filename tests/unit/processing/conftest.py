"""Shared pytest fixtures for document processing tests."""
# pylint: disable=redefined-outer-name

from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

import pytest

from src.models.processing import (
    DocumentElement,
    ProcessingResult,
    ProcessingStrategy,
)

# Chunking models removed; using Unstructured + LlamaIndex standard splitters.


@dataclass
class ChunkingParameters:
    """Parameters for chunking (test stub)."""

    chunk_size: int = 1000
    chunk_overlap: int = 100
    new_after_n_chars: int = 1200
    combine_text_under_n_chars: int = 200


# Document Processing Fixtures


@pytest.fixture
def mock_docmind_settings():
    """Mock DocMind settings for testing with realistic values."""
    settings = Mock()

    # Processing settings
    settings.processing.chunk_size = 1500
    settings.processing.chunk_overlap = 150
    settings.processing.new_after_n_chars = 1200
    settings.processing.combine_text_under_n_chars = 500
    settings.processing.multipage_sections = True

    # Cache settings
    settings.cache_dir = "./test_cache"
    settings.max_document_size_mb = 50

    # Performance settings
    settings.processing.batch_size = 10
    settings.processing.max_concurrent_documents = 5

    return settings


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return """
    # Document Title: Research Paper
    
    ## Abstract
    This paper presents a comprehensive analysis of document processing techniques
    using advanced natural language processing methods.
    
    ## Introduction
    Document processing has become increasingly important in modern applications.
    The ability to extract meaningful information from various document formats
    is crucial for many business and research applications.
    
    ## Methodology
    Our approach combines traditional parsing methods with machine learning
    techniques to achieve optimal results. We use a hybrid pipeline that
    processes documents through multiple stages.
    
    ## Results
    The experimental results demonstrate significant improvements in processing
    accuracy and speed compared to traditional methods. Our system achieved
    92% accuracy on the benchmark dataset.
    
    ## Conclusion
    This research contributes to the field of document processing by providing
    a novel approach that combines multiple techniques for optimal performance.
    """


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return """
    Welcome to DocMind AI
    
    DocMind AI is a revolutionary document processing system that leverages
    advanced artificial intelligence to extract insights from your documents.
    
    Key Features:
    - Multi-format support (PDF, DOCX, TXT, HTML, images)
    - Semantic chunking with intelligent boundary detection
    - Advanced embedding models for similarity search
    - Real-time processing with caching optimization
    
    Getting Started:
    1. Upload your documents
    2. Configure processing parameters
    3. Run the analysis
    4. Review the extracted insights
    
    For more information, please visit our documentation.
    """


@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Product Documentation</title>
    </head>
    <body>
        <h1>Product Overview</h1>
        <p>This is a comprehensive guide to our product features.</p>
        
        <h2>Installation</h2>
        <p>Follow these steps to install the software:</p>
        <ol>
            <li>Download the installer</li>
            <li>Run the setup wizard</li>
            <li>Configure your settings</li>
        </ol>
        
        <h2>Configuration</h2>
        <p>The system can be configured through the settings panel.</p>
        
        <table>
            <tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
            <tr><td>timeout</td><td>30s</td><td>Request timeout</td></tr>
            <tr><td>retries</td><td>3</td><td>Number of retry attempts</td></tr>
        </table>
    </body>
    </html>
    """


# Document Elements and Results


@pytest.fixture
def sample_document_elements():
    """Create sample DocumentElement objects with various categories."""
    return [
        DocumentElement(
            text="Research Paper: Advanced Document Processing",
            category="Title",
            metadata={
                "page_number": 1,
                "element_id": "title_1",
                "font_size": 24,
                "is_bold": True,
            },
        ),
        DocumentElement(
            text=(
                "This paper presents a comprehensive analysis of document "
                "processing techniques."
            ),
            category="NarrativeText",
            metadata={
                "page_number": 1,
                "element_id": "abstract_1",
                "section": "Abstract",
            },
        ),
        DocumentElement(
            text="Introduction",
            category="Title",
            metadata={
                "page_number": 2,
                "element_id": "title_2",
                "font_size": 18,
                "section_level": 2,
            },
        ),
        DocumentElement(
            text=(
                "Document processing has become increasingly important in modern "
                "applications."
            ),
            category="NarrativeText",
            metadata={
                "page_number": 2,
                "element_id": "intro_1",
                "section": "Introduction",
            },
        ),
        DocumentElement(
            text=(
                "The ability to extract meaningful information from documents "
                "is crucial."
            ),
            category="NarrativeText",
            metadata={
                "page_number": 2,
                "element_id": "intro_2",
                "section": "Introduction",
            },
        ),
        DocumentElement(
            text=(
                "| Method | Accuracy | Speed |\n| --- | --- | --- |\n| "
                "Traditional | 75% | Fast |\n| ML-based | 92% | Medium |"
            ),
            category="Table",
            metadata={
                "page_number": 3,
                "element_id": "table_1",
                "table_rows": 3,
                "table_cols": 3,
            },
        ),
        DocumentElement(
            text="Figure 1: Processing Pipeline Architecture",
            category="FigureCaption",
            metadata={
                "page_number": 4,
                "element_id": "figure_1",
                "associated_image": "pipeline_diagram.png",
            },
        ),
    ]


@pytest.fixture
def sample_processing_result(sample_document_elements):
    """Create a sample ProcessingResult for testing."""
    return ProcessingResult(
        elements=sample_document_elements,
        processing_time=2.5,
        strategy_used=ProcessingStrategy.HI_RES,
        metadata={
            "file_path": "/test/documents/sample.pdf",
            "file_size_mb": 1.2,
            "page_count": 4,
            "element_count": len(sample_document_elements),
            "extraction_method": "unstructured_io",
            "processing_timestamp": "2024-01-01T12:00:00Z",
        },
        document_hash="abc123def456",
    )


# File System Fixtures


@pytest.fixture
def test_documents_directory(tmp_path):
    """Create a temporary directory with test documents of various formats."""
    docs_dir = tmp_path / "test_documents"
    docs_dir.mkdir()

    # Create subdirectories
    (docs_dir / "pdfs").mkdir()
    (docs_dir / "texts").mkdir()
    (docs_dir / "images").mkdir()

    # Create test files with content
    test_files = {
        "research_paper.pdf": (
            "This is a sample PDF document about machine learning research."
        ),
        "user_manual.docx": (
            "User Manual\n\nThis manual provides instructions for using our software."
        ),
        "meeting_notes.txt": (
            "Meeting Notes - January 2024\n\nDiscussed project timeline and "
            "deliverables."
        ),
        "presentation.pptx": (
            "Slide 1: Project Overview\nSlide 2: Technical Architecture\nSlide 3: "
            "Implementation Plan"
        ),
        "webpage.html": (
            "<html><body><h1>Web Content</h1><p>This is web content.</p></body></html>"
        ),
        "document.rtf": "{\\rtf1 This is RTF content with formatting.}",
        "readme.md": "# Project README\n\n## Installation\n\nRun `pip install package`",
        "diagram.jpg": "JPEG image data (simulated)",
        "screenshot.png": "PNG image data (simulated)",
        "scan.tiff": "TIFF image data (simulated)",
        "unsupported.xyz": "This format is not supported by the processor",
    }

    # Create files in main directory
    for filename, content in test_files.items():
        (docs_dir / filename).write_text(content)

    # Create files in subdirectories
    (docs_dir / "pdfs" / "technical_spec.pdf").write_text(
        "Technical Specification Document"
    )
    (docs_dir / "texts" / "changelog.txt").write_text(
        "Version 1.0.0\n- Initial release"
    )
    (docs_dir / "images" / "logo.png").write_text("Company logo image data")

    return docs_dir


@pytest.fixture
def single_test_file(tmp_path, sample_pdf_content):
    """Create a single test file with PDF content."""
    test_file = tmp_path / "test_document.pdf"
    test_file.write_text(sample_pdf_content)
    return test_file


# Mock Objects


@pytest.fixture
def mock_unstructured_element():
    """Mock unstructured element with realistic properties."""
    element = Mock()
    element.text = "Sample text content from unstructured processing"
    element.category = "NarrativeText"

    # Mock metadata object
    element.metadata = Mock()
    element.metadata.page_number = 1
    element.metadata.element_id = "elem_123"
    element.metadata.parent_id = "parent_456"
    element.metadata.filename = "test_document.pdf"
    element.metadata.coordinates = (100, 200, 300, 400)  # x1, y1, x2, y2
    element.metadata.text_as_html = (
        "<p>Sample text content from unstructured processing</p>"
    )
    element.metadata.image_path = None

    return element


@pytest.fixture
def mock_llama_index_node():
    """Mock LlamaIndex node object."""
    node = Mock()
    node.text = "LlamaIndex node content"
    node.get_content.return_value = "LlamaIndex node content"
    node.metadata = {
        "element_category": "NarrativeText",
        "page_number": 1,
        "source_file": "/path/to/document.pdf",
        "processing_strategy": "hi_res",
        "element_index": 0,
    }

    # Mock node relationships
    node.relationships = {}
    node.excluded_embed_metadata_keys = []
    node.excluded_llm_metadata_keys = []
    node.metadata_seperator = "\n"
    node.metadata_template = "{metadata_str}"
    node.text_template = "{content}"

    return node


@pytest.fixture
def mock_document_processor():
    """Mock DocumentProcessor with realistic behavior."""
    processor = Mock()

    # Mock async methods
    processor.process_document_async = AsyncMock()
    processor.clear_cache = AsyncMock(return_value=True)
    processor.get_cache_stats = AsyncMock(
        return_value={
            "processor_type": "hybrid",
            "cache_hits": 10,
            "cache_misses": 5,
            "total_documents": 15,
        }
    )

    # Mock sync methods
    processor.get_strategy_for_file.return_value = ProcessingStrategy.HI_RES
    processor.override_config.return_value = None

    # Mock strategy mapping
    processor.strategy_map = {
        ".pdf": ProcessingStrategy.HI_RES,
        ".docx": ProcessingStrategy.HI_RES,
        ".txt": ProcessingStrategy.FAST,
        ".html": ProcessingStrategy.FAST,
        ".jpg": ProcessingStrategy.OCR_ONLY,
        ".png": ProcessingStrategy.OCR_ONLY,
    }

    return processor


@pytest.fixture
def mock_semantic_chunker():
    """Mock SemanticChunker with realistic behavior."""
    chunker = Mock()

    # Mock async methods
    chunker.chunk_elements_async = AsyncMock()

    # Mock sync methods
    chunker.optimize_parameters.return_value = ChunkingParameters()

    # Mock settings
    chunker.settings = Mock()
    chunker.default_parameters = ChunkingParameters()

    return chunker


# Performance Testing Fixtures


@pytest.fixture
def performance_test_data():
    """Create data for performance testing."""
    return {
        "small_document": "A" * 1000,  # 1KB
        "medium_document": "B" * 10000,  # 10KB
        "large_document": "C" * 100000,  # 100KB
        "huge_document": "D" * 1000000,  # 1MB
    }


@pytest.fixture
def batch_test_documents(tmp_path, performance_test_data):
    """Create a batch of test documents for performance testing."""
    batch_dir = tmp_path / "batch_test"
    batch_dir.mkdir()

    document_paths = []
    for _i, (size_name, content) in enumerate(performance_test_data.items()):
        for j in range(3):  # 3 documents per size category
            file_path = batch_dir / f"{size_name}_{j}.txt"
            file_path.write_text(content)
            document_paths.append(file_path)

    return document_paths


# Error Testing Fixtures


@pytest.fixture
def corrupted_test_files(tmp_path):
    """Create test files that simulate various error conditions."""
    error_dir = tmp_path / "error_tests"
    error_dir.mkdir()

    # Empty file
    (error_dir / "empty.pdf").write_text("")

    # File with invalid characters (simulation)
    (error_dir / "invalid_chars.txt").write_bytes(b"\x00\x01\x02\x03Invalid content")

    # Very large file name
    long_name = "a" * 200 + ".pdf"
    (error_dir / long_name).write_text("Content with very long filename")

    # File with special characters
    (error_dir / "special@#$.pdf").write_text(
        "Content with special characters in filename"
    )

    return error_dir


# Integration Test Fixtures


@pytest.fixture
def realistic_document_collection(tmp_path):
    """Create a realistic collection of documents for integration testing."""
    collection_dir = tmp_path / "realistic_collection"
    collection_dir.mkdir()

    # Academic paper
    academic_paper = """
    # Machine Learning in Document Processing: A Comprehensive Review
    
    ## Abstract
    This paper reviews the current state of machine learning applications
    in document processing, covering both traditional and deep learning approaches.
    
    ## 1. Introduction
    Document processing has evolved significantly with the advent of machine learning.
    Traditional rule-based systems are being replaced by more sophisticated ML models.
    
    ## 2. Literature Review
    Recent work by Smith et al. (2023) demonstrated significant improvements using
    transformer architectures. Similarly, Jones and Wilson (2024) showed promising
    results with multimodal approaches.
    
    ## 3. Methodology
    We employed a hybrid approach combining convolutional neural networks for
    visual feature extraction with recurrent networks for sequence modeling.
    
    ## 4. Results
    Our experiments on the DocBench dataset achieved 94.7% accuracy, representing
    a 15% improvement over previous state-of-the-art methods.
    
    ## 5. Conclusion
    The integration of machine learning techniques in document processing shows
    great promise for future applications.
    """

    # Business report
    business_report = """
    QUARTERLY BUSINESS REPORT - Q4 2024
    
    Executive Summary
    - Revenue increased 23% year-over-year
    - Customer satisfaction improved to 4.8/5.0
    - New product line launched successfully
    
    Financial Performance
    Total Revenue: $12.5M (+23% YoY)
    Gross Margin: 67% (+5% YoY)
    Net Income: $3.2M (+31% YoY)
    
    Key Achievements
    1. Launched AI-powered document processing platform
    2. Expanded to 3 new international markets
    3. Increased team size by 40%
    
    Looking Forward
    Q1 2025 will focus on scaling operations and enhancing
    our core technology platform.
    """

    # Technical manual
    technical_manual = """
    API Documentation - Document Processing Service
    
    Authentication
    All API requests require authentication via API key:
    Authorization: Bearer your-api-key
    
    Endpoints
    
    POST /api/v1/documents/upload
    Upload a document for processing
    
    Parameters:
    - file: Document file (PDF, DOCX, TXT)
    - strategy: Processing strategy (hi_res, fast, ocr_only)
    - options: Additional processing options
    
    Response:
    {
      "document_id": "uuid",
      "status": "processing",
      "estimated_time": 30
    }
    
    GET /api/v1/documents/{id}/status
    Check processing status
    
    Response:
    {
      "document_id": "uuid",
      "status": "completed",
      "elements": [...],
      "processing_time": 25.3
    }
    
    Error Codes
    400: Bad Request - Invalid parameters
    401: Unauthorized - Invalid API key
    429: Too Many Requests - Rate limit exceeded
    500: Internal Server Error - Processing failed
    """

    # Create files
    (collection_dir / "academic_paper.pdf").write_text(academic_paper)
    (collection_dir / "business_report.docx").write_text(business_report)
    (collection_dir / "technical_manual.txt").write_text(technical_manual)

    return collection_dir


# Utility Functions for Tests


@pytest.fixture
def test_utils():
    """Provide utility functions for testing."""

    class TestUtils:
        """Utility helpers for tests."""

        @staticmethod
        def create_mock_processing_result(
            element_count: int = 5,
            processing_time: float = 1.5,
            strategy: ProcessingStrategy = ProcessingStrategy.FAST,
        ) -> ProcessingResult:
            """Create a mock ProcessingResult with specified parameters."""
            elements = [
                DocumentElement(
                    text=f"Element {i} content",
                    category="NarrativeText",
                    metadata={"element_index": i},
                )
                for i in range(element_count)
            ]

            return ProcessingResult(
                elements=elements,
                processing_time=processing_time,
                strategy_used=strategy,
                metadata={"test": True},
                document_hash=f"hash_{element_count}",
            )

        @staticmethod
        def assert_processing_result_valid(result: ProcessingResult):
            """Assert that a ProcessingResult is valid."""
            assert isinstance(result, ProcessingResult)
            assert isinstance(result.elements, list)
            assert all(isinstance(elem, DocumentElement) for elem in result.elements)
            assert result.processing_time >= 0
            assert isinstance(result.strategy_used, ProcessingStrategy)
            assert isinstance(result.metadata, dict)
            assert isinstance(result.document_hash, str)

        @staticmethod
        def assert_chunking_result_valid(result):
            """Assert that a ChunkingResult is valid."""
            assert hasattr(result, "chunks")
            assert hasattr(result, "total_elements")
            assert hasattr(result, "boundary_accuracy")
            assert hasattr(result, "processing_time")
            assert hasattr(result, "parameters")

            assert isinstance(result.chunks, list)
            assert result.total_elements >= 0
            assert 0.0 <= result.boundary_accuracy <= 1.0
            assert result.processing_time >= 0

    return TestUtils
