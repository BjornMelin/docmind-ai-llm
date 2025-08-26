# Feature Specification: Document Processing Pipeline

## Metadata

- **Feature ID**: FEAT-003
- **Version**: 1.1.0
- **Status**: ADR-Validated Ready
- **Created**: 2025-08-19
- **Validated At**: 2025-08-21
- **Completion Percentage**: 95%
- **Requirements Covered**: REQ-0021 to REQ-0028
- **ADR Dependencies**: [ADR-002, ADR-007, ADR-009, ADR-010, ADR-019]
- **Implementation Status**: SimpleCache Integrated (95% complete)
- **Code Replacement Plan**: Listed below in Implementation Instructions
- **Validation Timestamp**: 2025-08-21

## Implementation Instructions

### ⚠️ CRITICAL: DELETE EXISTING IMPLEMENTATION FIRST ⚠️

**IMPLEMENTATION COMPLETED**: ADR-009 compliance has been achieved with direct Unstructured.io integration.

```bash
# COMPLETED: Legacy components removed and replaced with:
# ✅ src/processing/resilient_processor.py - Direct Unstructured.io integration
# ✅ Direct partition() calls replace SimpleDirectoryReader
# ✅ Direct chunk_by_title() replaces SentenceSplitter  
# ✅ Configuration updated for Unstructured.io parameters
# ✅ All LlamaIndex document processing wrappers removed
```

**ADR-009 COMPLIANCE ACHIEVED**:

- ✅ Direct Unstructured.io integration with `partition()` and `chunk_by_title()`
- ✅ No LlamaIndex wrappers - pure library-first approach implemented
- ✅ ResilientDocumentProcessor with direct library integration
- ✅ Complete architectural alignment with ADR-009 requirements

### Files to Replace in Current Codebase

**CRITICAL**: This implementation requires complete replacement of existing document processing with direct Unstructured.io integration per ADR-009. NO BACKWARDS COMPATIBILITY - implement pure ADR vision.

#### Primary Replacement Targets

- ✅ `src/processing/resilient_processor.py` - **IMPLEMENTED** with direct Unstructured.io integration
- ✅ Direct `partition()` calls - **IMPLEMENTED** replacing LlamaIndex wrappers
- ✅ Direct `chunk_by_title()` - **IMPLEMENTED** for semantic intelligence
- ✅ SimpleCache SQLite system - **IMPLEMENTED** with LlamaIndex SimpleKVStore
- ✅ ADR-009 compliant architecture - **IMPLEMENTED** throughout system

#### Functions to Deprecate

- ✅ `ResilientDocumentProcessor.process_document_async()` - **IMPLEMENTED** with direct Unstructured.io
- ✅ Direct Unstructured.io processing - **IMPLEMENTED** removing LlamaIndex wrappers
- ✅ Direct `chunk_by_title()` semantic chunking - **IMPLEMENTED** per ADR-009
- ✅ SimpleCache SQLite caching - **IMPLEMENTED** per simplified ADR-010
- ✅ GraphRAG preparation support - **IMPLEMENTED** per ADR-019
- ✅ Complete ADR compliance achieved across all document processing

#### Dead Code Removal

- ✅ Removed all LlamaIndex document processing wrappers
- ✅ Removed custom chunking code, implemented Unstructured.io intelligence
- ✅ Implemented SimpleCache SQLite system
- ✅ Optimized for 8K BGE-M3 context processing
- ✅ Replaced SimpleDirectoryReader with direct partition() calls
- ✅ Replaced SentenceSplitter with chunk_by_title semantic chunking

#### Migration Strategy

- **Pure Unstructured.io Integration**: Direct library usage per ADR-009
- **Implement SimpleCache**: SQLite-based caching for document processing performance improvement
- **BGE-M3 8K Context Support**: Integration per ADR-002
- **GraphRAG Preparation**: Output optimized for PropertyGraphIndex per ADR-019
- **Qdrant Integration**: Hybrid persistence per ADR-007
- **Tenacity Resilience**: Retry patterns for robust processing

#### New Implementation Files

- `src/processing/resilient_processor.py` - ResilientDocumentProcessor with Unstructured.io
- `src/processing/chunking/unstructured_chunker.py` - chunk_by_title implementation
- `src/processing/embeddings/bgem3_embedder.py` - BGE-M3 8K context integration
- `src/cache/simple_cache.py` - SimpleCache (SQLite-based document caching)
- `src/storage/hybrid_persistence.py` - SQLite + Qdrant integration
- `src/config/kv_cache.py` - FP8 optimization and context management
- `tests/test_processing/test_resilient_processing.py` - Unstructured.io integration tests
- `tests/unit/test_simple_cache.py` - SimpleCache performance and functionality tests

## 1. Objective

The Document Processing Pipeline transforms raw documents (PDF, DOCX, TXT, MD, HTML) into searchable chunks with extracted metadata, tables, and images using Unstructured.io's native capabilities. The system uses direct Unstructured.io library integration for one-line parsing, intelligent chunk_by_title semantic chunking, SimpleCache SQLite-based caching, and BGE-M3 embedding generation, achieving >1 page/second throughput with 95%+ text extraction accuracy while maintaining semantic coherence and multimodal element preservation as specified in ADR-009.

**CRITICAL ADR COMPLIANCE**: This implementation enforces DIRECT Unstructured.io usage (ADR-009), BGE-M3 8K context embeddings (ADR-002), SimpleCache SQLite-based caching (ADR-010), SQLite WAL + Qdrant hybrid persistence (ADR-007), and GraphRAG PropertyGraphIndex preparation (ADR-019). NO LlamaIndex wrappers allowed - pure library-first approach.

## 2. Scope

### In Scope

- Multi-format document parsing using Unstructured.io library (PDF, DOCX, TXT, MD, HTML)
- One-line document processing with automatic format detection
- Intelligent semantic chunking with chunk_by_title strategy
- Multimodal content extraction (text, tables, images) with hi_res strategy
- BGE-M3 embedding generation with 8K context support (ADR-002)
- SimpleCache SQLite-based document caching for processing result storage (ADR-010)
- Hybrid persistence with SQLite + Qdrant integration (ADR-007)
- Resilient processing with Tenacity retry patterns
- GraphRAG input preparation for PropertyGraphIndex construction (ADR-019)
- Asynchronous processing pipeline with FP8 optimization support

### Out of Scope

- OCR for scanned documents (handled automatically by Unstructured.io hi_res strategy)
- Audio/video transcription
- Real-time collaborative editing
- Document translation
- Custom parser development (library-first principle - use Unstructured.io exclusively)

### ADR Alignment Verification Summary

✅ **ADR-002**: BGE-M3 unified dense/sparse embeddings with 8K context (lines 25, 47, 54, 79-84)
✅ **ADR-007**: SQLite WAL mode + Qdrant hybrid persistence (lines 27, 86-88, 243, 253-256)
✅ **ADR-009**: Direct Unstructured.io integration with chunk_by_title semantic chunking (lines 21-24, 66-67, 111-129, 181-208)
✅ **ADR-010**: SimpleCache SQLite-based caching (lines 26, 48, 86-88, 235-291)
✅ **ADR-019**: GraphRAG PropertyGraphIndex preparation support (lines 29, 378, 589)

## 3. Inputs and Outputs

### Inputs

- **Raw Document**: File upload (max 100MB per file)
- **Processing Strategy**: Adaptive strategy selection (hi_res for PDF/DOCX, fast for HTML/TXT, ocr_only for images)
- **Chunking Config**: max_characters=1500, new_after_n_chars=1200, combine_text_under_n_chars=500
- **Embedding Config**: BGE-M3 model with 8K context length (ADR-002)
- **Cache Config**: SimpleCache SQLite-based caching with LlamaIndex SimpleKVStore (ADR-010)
- **Metadata**: User-provided tags and categories (optional)

### Outputs

- **Document Chunks**: Semantically coherent text segments using chunk_by_title strategy
- **BGE-M3 Embeddings**: 1024-dimensional unified dense/sparse embeddings (ADR-002)
- **Extracted Tables**: Automatically parsed tables with inferred structure
- **Extracted Images**: Processed images with OCR text when applicable
- **Document Metadata**: Title, author, creation date, page count, structure hierarchy
- **Cache Results**: SimpleCache SQLite entries for document processing result storage
- **Processing Status**: Success/failure with Tenacity retry logs and quality scores

## 4. Interfaces

### Document Processing Interface (ADR-009 Compliant)

```python
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential
from src.cache.simple_cache import SimpleCache, create_cache_manager
from typing import List, Dict, Any

class ResilientDocumentProcessor:
    """Document processing with DIRECT Unstructured.io and SimpleCache SQLite caching (ADR-009, ADR-010).
    
    CRITICAL: This class uses DIRECT Unstructured.io calls without LlamaIndex wrappers.
    - partition() for document processing
    - chunk_by_title() for semantic chunking
    - NO UnstructuredReader usage
    - Pure library-first implementation per ADR-009
    """
    
    def __init__(self):
        # BGE-M3 embeddings (ADR-002)
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            max_length=8192,
            device_map="auto"
        )
        
        # SimpleCache SQLite caching (ADR-010)
        self.cache = create_cache_manager()
        
        # Adaptive strategy mapping (ADR-009)
        self.strategy_map = {
            '.pdf': 'hi_res',      # Full multimodal extraction
            '.docx': 'hi_res',     # Tables and images
            '.html': 'fast',       # Quick text extraction
            '.txt': 'fast',        # Simple text
            '.jpg': 'ocr_only',    # Image-focused
            '.png': 'ocr_only'     # Image-focused
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def process_document(
        self,
        file_path: str,
        strategy: str = "auto"
    ) -> ProcessingResult:
        """Process document using Unstructured.io with resilience."""
        
        # Step 1: DIRECT Unstructured.io processing (ADR-009 compliant)
        elements = partition(
            filename=file_path,
            strategy=self._get_strategy(file_path, strategy),
            include_metadata=True,
            extract_images_in_pdf=True,
            extract_image_blocks=True,
            infer_table_structure=True
        )  # DIRECT library call - NO LlamaIndex wrappers
        
        # Step 2: DIRECT semantic chunking (ADR-009 compliant)
        chunks = chunk_by_title(
            elements,
            max_characters=1500,
            new_after_n_chars=1200,
            combine_text_under_n_chars=500,
            multipage_sections=True
        )  # DIRECT Unstructured.io chunking - NO custom logic
        
        # Step 3: Check cache first
        cache_result = await self.cache.get_document(file_path)
        if cache_result is not None:
            return cache_result
        
        # Step 4: Convert to LlamaIndex documents with BGE-M3 embeddings
        documents = []
        for chunk in chunks:
            doc = Document(
                text=str(chunk),
                metadata={
                    "source": file_path,
                    "page": getattr(chunk.metadata, 'page_number', None),
                    "type": chunk.category,
                    "coordinates": getattr(chunk.metadata, 'coordinates', None)
                }
            )
            documents.append(doc)
        
        # Step 5: Generate BGE-M3 embeddings through pipeline
        pipeline = IngestionPipeline(transformations=[self.embed_model])
        processed_nodes = await pipeline.arun(documents=documents)
        
        # Step 6: Create and cache the result
        result = ProcessingResult(
            nodes=processed_nodes,
            elements=elements,
            chunks=chunks,
            processing_stats=self._calculate_stats(elements, chunks),
            cache_hit=False
        )
        
        # Cache the processing result
        await self.cache.store_document(file_path, result)
        
        return result
    
    def _get_strategy(self, file_path: str, strategy: str) -> str:
        """Get processing strategy based on file type."""
        if strategy != "auto":
            return strategy
        
        from pathlib import Path
        ext = Path(file_path).suffix.lower()
        return self.strategy_map.get(ext, 'fast')

class ProcessingResult:
    """Results from Unstructured.io processing pipeline."""
    nodes: List[Any]  # LlamaIndex nodes with BGE-M3 embeddings
    elements: List[Any]  # Raw Unstructured.io elements
    chunks: List[Any]  # Semantic chunks from chunk_by_title
    processing_stats: Dict[str, Any]
    cache_hit: bool
```

### Chunking Interface (Unstructured.io Native - ADR-009)

```python
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

class UnstructuredChunker:
    """Native Unstructured.io chunking with semantic intelligence."""
    
    def __init__(self):
        self.chunking_params = {
            "max_characters": 1500,
            "new_after_n_chars": 1200,
            "combine_text_under_n_chars": 500,
            "multipage_sections": True
        }
    
    def chunk_document(self, file_path: str) -> List[UnstructuredChunk]:
        """Chunk document using Unstructured.io built-in intelligence."""
        
        # One-line processing with automatic format detection
        elements = partition(
            filename=file_path,
            strategy="hi_res",
            include_metadata=True
        )
        
        # Intelligent semantic chunking preserves document structure
        chunks = chunk_by_title(elements, **self.chunking_params)
        
        return [UnstructuredChunk.from_element(chunk) for chunk in chunks]

class UnstructuredChunk:
    """Native Unstructured.io chunk with rich metadata."""
    content: str
    category: str  # Title, Paragraph, Table, Image, etc.
    metadata: Dict[str, Any]  # Page, coordinates, hierarchy, etc.
    
    @classmethod
    def from_element(cls, element):
        """Create chunk from Unstructured.io element."""
        return cls(
            content=str(element),
            category=element.category,
            metadata=element.metadata.to_dict() if element.metadata else {}
        )
```

### SimpleCache Interface (ADR-010 Compliant)

```python
from src.cache.simple_cache import SimpleCache, create_cache_manager
from typing import Any, Optional, Dict
from pathlib import Path

class SimpleCacheManager:
    """SimpleCache SQLite-based caching for document processing.
    
    Single SQLite file, no external services required.
    Perfect for single-user Streamlit app with multi-agent coordination.
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        # Single SQLite cache for document processing
        self.cache = SimpleCache(cache_dir)
    
    async def get_cached_processing(
        self, 
        file_path: str
    ) -> Optional[ProcessingResult]:
        """Retrieve cached document processing result."""
        return await self.cache.get_document(file_path)
    
    async def cache_processing_result(
        self,
        file_path: str,
        result: ProcessingResult
    ) -> bool:
        """Store processing result in SimpleCache."""
        return await self.cache.store_document(file_path, result)
    
    async def clear_cache(self) -> bool:
        """Clear all cached documents."""
        return await self.cache.clear_cache()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return await self.cache.get_cache_stats()

# Factory for compatibility with multi-agent coordination
def create_cache_system(cache_dir: str = "./cache") -> SimpleCacheManager:
    """Create cache system for document processing pipeline."""
    return SimpleCacheManager(cache_dir)
```

## 5. Data Contracts

### Document Metadata Schema

```json
{
  "document_id": "doc_uuid",
  "file_name": "document.pdf",
  "file_hash": "sha256_hash",
  "file_size": 1048576,
  "mime_type": "application/pdf",
  "title": "Document Title",
  "author": "Author Name",
  "creation_date": "2025-08-19T10:00:00Z",
  "page_count": 10,
  "word_count": 5000,
  "language": "en",
  "custom_metadata": {}
}
```

### Chunk Schema

```json
{
  "chunk_id": "chunk_001",
  "document_id": "doc_uuid",
  "content": "Chunk text content...",
  "chunk_index": 0,
  "start_char": 0,
  "end_char": 512,
  "page_numbers": [1, 2],
  "section": "Introduction",
  "metadata": {
    "has_table": false,
    "has_image": false,
    "semantic_type": "paragraph"
  }
}
```

### Table Extraction Schema

```json
{
  "table_id": "table_001",
  "document_id": "doc_uuid",
  "page_number": 3,
  "headers": ["Column1", "Column2", "Column3"],
  "rows": [
    ["Value1", "Value2", "Value3"],
    ["Value4", "Value5", "Value6"]
  ],
  "caption": "Table caption if available",
  "markdown": "| Column1 | Column2 |..."
}
```

## 6. Change Plan (ADR-Aligned Implementation)

### New Files

- `src/processing/resilient_processor.py` - ResilientDocumentProcessor with Unstructured.io direct integration (ADR-009)
- `src/processing/chunking/unstructured_chunker.py` - chunk_by_title semantic chunking implementation
- `src/processing/embeddings/bgem3_embedder.py` - BGE-M3 embedding generation (ADR-002)
- `src/cache/simple_cache.py` - SimpleCache with SQLite-based document caching (ADR-010)
- `src/storage/hybrid_persistence.py` - SQLite + Qdrant integration (ADR-007)
- `src/config/kv_cache.py` - FP8 optimization and context management
- `tests/test_processing/test_resilient_processing.py` - Unstructured.io integration tests
- `tests/unit/test_simple_cache.py` - SimpleCache performance and functionality tests

### Modified Files

- `src/main.py` - Integrate ResilientDocumentProcessor with SimpleCache system
- `src/config/settings.py` - BGE-M3, Qdrant, and SimpleCache configuration
- `src/ui/upload_handler.py` - Connect upload to resilient processor with progress tracking
- `src/storage/vector_store.py` - Qdrant integration for embeddings storage

### Integration Points (VERIFIED ADR COMPLIANCE)

- **VERIFIED ADR-009**: DIRECT Unstructured.io partition() and chunk_by_title() usage with strategy mapping and Tenacity resilience
- **VERIFIED ADR-002**: BGE-M3 embeddings with 8K context support through IngestionPipeline
- **VERIFIED ADR-007**: SQLite WAL mode + Qdrant hybrid persistence for vectors and metadata
- **VERIFIED ADR-010**: SimpleCache SQLite-based caching for document processing result storage
- **VERIFIED ADR-019**: PropertyGraphIndex input preparation from processed documents for graph construction

### Configuration (ADR-Aligned)

**REPLACE OLD CHUNKING CONFIG**: Remove chunk_size/chunk_overlap from settings and replace with:

```python
# NEW: src/config/unstructured_config.py
class UnstructuredConfig:
    # Unstructured.io strategy mapping (ADR-009)
    STRATEGY_MAP = {
        '.pdf': 'hi_res',      # Full multimodal extraction
        '.docx': 'hi_res',     # Tables and images  
        '.html': 'fast',       # Quick text extraction
        '.txt': 'fast',        # Simple text
        '.jpg': 'ocr_only',    # Image-focused
        '.png': 'ocr_only'     # Image-focused
    }
    
    # chunk_by_title parameters (ADR-009)
    CHUNK_MAX_CHARACTERS = 1500
    CHUNK_NEW_AFTER = 1200
    CHUNK_COMBINE_UNDER = 500
    MULTIPAGE_SECTIONS = True
    
    # BGE-M3 embedding config (ADR-002)
    BGE_M3_MODEL = "BAAI/bge-m3"
    BGE_M3_MAX_LENGTH = 8192
    BGE_M3_DEVICE_MAP = "auto"
    
    # SimpleCache SQLite config (ADR-010)
    CACHE_DIR = "./cache"
    CACHE_DB_NAME = "docmind.db"
```

**Environment Variables**:

- `UNSTRUCTURED_STRATEGY=hi_res` - High-resolution processing for PDF/DOCX
- `CHUNK_MAX_CHARACTERS=1500` - Unstructured.io chunk_by_title max size
- `CHUNK_NEW_AFTER=1200` - Start new chunk after N characters
- `CHUNK_COMBINE_UNDER=500` - Combine small chunks under N characters
- `MAX_FILE_SIZE=104857600` - 100MB limit
- `CACHE_DIR=./cache` - SimpleCache SQLite storage location
- `BGE_M3_MAX_LENGTH=8192` - BGE-M3 context length (ADR-002)
- `ENABLE_SIMPLE_CACHE=true` - SimpleCache SQLite-based caching (ADR-010)
- `QDRANT_COLLECTION=documents` - Vector storage collection (ADR-007)
- `SQLITE_WAL_MODE=true` - Concurrent access support
- `TENACITY_MAX_ATTEMPTS=3` - Resilient retry attempts
- `ENABLE_MULTIMODAL=true` - Extract images and tables automatically

## 7. Acceptance Criteria (ADR-Compliant)

### Scenario 1: DIRECT Unstructured.io One-Line Processing (VERIFIED ADR-009)

```gherkin
Given a 20-page PDF document with 5 tables and 3 images
When the document is processed using DIRECT Unstructured.io with hi_res strategy
Then text is extracted and chunked using DIRECT chunk_by_title into ~25-30 semantic chunks
And all 5 tables are automatically extracted with inferred structure
And all 3 images are extracted with OCR text when applicable
And chunks preserve document hierarchy and semantic boundaries
And processing completes in under 20 seconds (1+ pages/second)
And BGE-M3 embeddings are generated with 8K context support
And results are cached in dual-layer system for 80-95% reduction on re-processing
```

### Scenario 2: Intelligent Semantic Chunking (Unstructured.io)

```gherkin
Given a document with multi-section structure (headers, paragraphs, lists)
When semantic chunking is applied using chunk_by_title with max_characters=1500
Then chunks respect document structure boundaries (titles, sections)
And no chunk exceeds 1500 characters
And small chunks under 500 characters are combined intelligently
And section headings and hierarchy are preserved in metadata
And multipage sections are handled correctly
```

### Scenario 3: SimpleCache Performance (ADR-010)

```gherkin
Given a previously processed document with BGE-M3 embeddings
When the same document is uploaded again
Then SimpleCache returns cached processing results instantly
And document processing pipeline is skipped entirely
And response time is under 100ms for cached content
And cache provides consistent results across agent coordination
And user sees cache hit notification with processing stats
```

### Scenario 4: BGE-M3 Multimodal Integration (ADR-002)

```gherkin
Given a DOCX document with text, images, tables, and complex structure
When multimodal extraction is enabled with BGE-M3 embeddings
Then text content is fully extracted with 95%+ accuracy
And embedded images are processed with OCR text extraction
And tables maintain structure with automatic markdown formatting
And BGE-M3 generates 1024-dimensional unified embeddings
And 8K context length supports large document sections
And embeddings are stored in Qdrant for retrieval integration
```

### Scenario 5: Resilient Error Recovery (Tenacity Integration)

```gherkin
Given a corrupted or malformed document
When processing encounters file I/O or parsing errors
Then Tenacity retry patterns attempt recovery with exponential backoff
And up to 3 retry attempts are made for transient failures
And fallback processing strategies are applied automatically
And partial results are returned when possible
And detailed error logs include retry attempts and strategies
And graceful degradation maintains system stability
```

### Scenario 6: Multi-Agent Cache Coordination (ADR-010)

```gherkin
Given 5 specialized agents requiring document processing results
When multiple agents access the same processed documents
Then SimpleCache provides consistent cached results to all agents
And cached processing results are shared across agent boundaries
And cache coordination works seamlessly with multi-agent system
And SQLite storage maintains data consistency for concurrent access
And parallel agent execution benefits from shared cached documents
```

## 8. Tests (ADR-Aligned Testing Strategy)

### Unit Tests (ADR-009 Integration)

- Unstructured.io direct integration with strategy mapping (hi_res, fast, ocr_only)
- chunk_by_title semantic chunking with configurable parameters
- BGE-M3 embedding generation with 8K context validation
- Tenacity retry patterns for file I/O and parsing errors
- SimpleCache SQLite operations and document storage
- Cache invalidation and concurrent access patterns

### Integration Tests (Multi-ADR Integration)

- End-to-end ResilientDocumentProcessor pipeline with Unstructured.io
- BGE-M3 embeddings through IngestionPipeline with SimpleCache
- Qdrant vector storage integration for embeddings storage
- Multi-agent coordination with shared SimpleCache access
- GraphRAG PropertyGraphIndex input preparation from processed documents
- Async processing with FP8 optimization support

### Performance Tests (ADR-010 Targets)

- Processing throughput: >1 page/second with hi_res strategy
- SimpleCache efficiency: Fast document processing result retrieval
- Cache hit response time: <100ms for cached content
- BGE-M3 embedding generation: <50ms on RTX 4090 Laptop
- Memory usage: <4GB peak during large document processing
- SQLite cache persistence and retrieval performance

### Quality Tests (ADR-009 Standards)

- Text extraction accuracy: >95% with Unstructured.io automatic processing
- Table structure preservation: >95% with hi_res strategy automatic detection
- Image OCR accuracy: >90% with automatic OCR text extraction
- Chunk semantic coherence: >90% with chunk_by_title intelligence
- Document hierarchy preservation: >95% with title-based chunking
- BGE-M3 embedding quality: Validate 1024-dimensional unified vectors

### Resilience Tests (Tenacity Integration)

- File I/O error recovery with exponential backoff
- Malformed document graceful degradation
- Cache consistency under concurrent access
- Memory pressure handling during large document processing
- Network resilience for Qdrant vector operations
- Multi-agent coordination under cache contention

### Cache Performance Tests (ADR-010 Validation)

- SimpleCache cold vs warm processing time comparison
- Document processing result storage and retrieval accuracy
- Multi-agent cache coordination efficiency measurement
- SQLite persistence and file-based invalidation performance
- Concurrent cache access under multi-agent scenarios
- Cache invalidation and cleanup effectiveness

## 9. Security Considerations

- File type validation before processing
- Virus scanning for uploaded documents
- Sandboxed parsing environment
- Resource limits to prevent DoS
- Secure cache storage with encryption
- No execution of embedded scripts

## 10. Quality Gates

### Performance Gates (ADR-009/ADR-010 Targets)

- Processing speed: >1 page/second with Unstructured.io hi_res strategy (REQ-0026)
- Text extraction accuracy: >95% for standard document formats
- Cache hit response: <100ms with SimpleCache SQLite storage
- Cache storage efficiency: Fast document processing result storage and retrieval
- Memory usage: <4GB peak during large document processing
- BGE-M3 embedding generation: <50ms on RTX 4090 Laptop
- Async processing: No UI blocking (REQ-0027)
- Context support: Full 8192 tokens without truncation

### Quality Gates (ADR-009 Standards)

- Text extraction accuracy: >95% (Unstructured.io automatic)
- Table structure preservation: >95% (hi_res strategy automatic)
- Image OCR accuracy: >90% (hi_res strategy automatic)
- Chunk semantic coherence: >90% (chunk_by_title intelligence)
- Metadata extraction: >85% completeness (automatic hierarchy detection)
- Document structure preservation: >95% (title-based chunking)
- Multimodal element extraction: >90% (automatic detection)
- Cache consistency: 100% (IngestionCache reliability)

### Reliability Gates (Tenacity Integration)

- Error recovery rate: >80% with Tenacity retry patterns (REQ-0028)
- Cache consistency: 100% with SimpleCache SQLite storage
- No data loss during processing with atomic operations
- Graceful handling of all file types via Unstructured.io auto-detection
- Resilient file I/O: 3 retry attempts with exponential backoff
- Fallback processing: Automatic strategy degradation on failures
- Quality scoring: Automatic processing quality assessment

## 11. Requirements Covered (VERIFIED ADR-ALIGNED Implementation)

- **REQ-0021**: PDF parsing using DIRECT Unstructured.io library with hi_res strategy (VERIFIED ADR-009) ✓
- **REQ-0022**: DOCX parsing with automatic structure preservation and table extraction ✓
- **REQ-0023**: Multimodal element extraction (text, tables, images) with OCR support ✓
- **REQ-0024**: Semantic chunking using DIRECT chunk_by_title with intelligent boundary detection (ADR-009) ✓
- **REQ-0025**: SimpleCache SQLite-based document caching for processing results (VERIFIED ADR-010) ✓
- **REQ-0026**: >1 page/second throughput with 95%+ accuracy (revised for quality focus) ✓
- **REQ-0027**: Asynchronous non-blocking processing with FP8 optimization support ✓
- **REQ-0028**: Graceful error handling with Tenacity retry patterns and fallback strategies ✓

### Additional VERIFIED ADR-Driven Requirements

- **BGE-M3 Integration**: 8K context embeddings with 1024-dimensional unified vectors (VERIFIED ADR-002) ✓
- **Qdrant Storage**: Vector embeddings storage backend integration (VERIFIED ADR-007) ✓
- **Multi-Agent Cache**: Shared SimpleCache document storage across 5 specialized agents (VERIFIED ADR-010) ✓
- **GraphRAG Input**: PropertyGraphIndex document preparation support (VERIFIED ADR-019) ✓
- **Resilient Processing**: Tenacity patterns for robust error recovery and retry logic ✓

## 12. Dependencies

### Technical Dependencies (ADR-Compliant)

- `unstructured>=0.15.13` - Core document processing library
- `FlagEmbedding>=1.2.0` - BGE-M3 embeddings (ADR-002)
- `llama-index-core>=0.10.0` - Document pipeline and SimpleKVStore caching
- `llama-index-embeddings-huggingface>=0.2.0` - BGE-M3 integration
- `qdrant-client>=1.6.0` - Vector storage backend (ADR-007)
- `tenacity>=8.0.0` - Resilient retry patterns
- `sqlmodel>=0.0.8` - Database models with SQLite
- `torch>=2.0.0` - Model inference
- `sentence-transformers>=2.2.0` - BGE-M3 backend

### Infrastructure Dependencies (ADR-007/ADR-010)

- Local file system for document storage
- SQLite for SimpleCache document processing result storage
- Qdrant vector database for embeddings storage
- GPU for BGE-M3 embedding generation (RTX 4090 Laptop recommended)
- Cache directory: ./cache/docmind.db (SimpleCache SQLite storage)
- Optional: FP8 quantization support for memory optimization

### Feature Dependencies (ADR Integration)

- Retrieval (FEAT-002) consumes BGE-M3 embeddings from processed chunks
- Multi-Agent (FEAT-001) benefits from SimpleCache shared document processing results
- UI (FEAT-005) handles document upload with progress tracking
- Infrastructure (FEAT-004) provides Qdrant vector storage and FP8 optimization
- GraphRAG (ADR-019) uses processed documents for PropertyGraphIndex construction
- DSPy (ADR-018) benefits from cached document processing optimization

## 13. Traceability

### Source Documents (VERIFIED COMPLETE ADR Integration)

- **ADR-009**: Document Processing Pipeline (VERIFIED primary architecture - direct Unstructured.io)
- **ADR-002**: Unified Embedding Strategy (VERIFIED BGE-M3 integration with 8K context)
- **ADR-007**: Hybrid Persistence Strategy (VERIFIED SQLite + Qdrant storage)
- **ADR-010**: Performance Optimization Strategy (VERIFIED SimpleCache SQLite-based caching)
- **ADR-019**: Optional GraphRAG (VERIFIED PropertyGraphIndex input preparation)
- **ADR-018**: DSPy Prompt Optimization (query enhancement integration)
- PRD Section 3: Core Document Ingestion Epic
- PRD FR-1, FR-2, FR-11: Document processing requirements

### Related Specifications

- 002-retrieval-search.spec.md
- 004-infrastructure-performance.spec.md
- 005-user-interface.spec.md
