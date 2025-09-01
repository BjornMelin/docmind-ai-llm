"""Sample document fixtures for integration testing.

This module provides realistic test documents and fixtures for comprehensive
integration testing of document processing, embedding, and retrieval workflows.

Test Documents:
- PDF research paper (simulated via text)
- Technical documentation (Markdown)
- Business report (structured text)
- Code documentation (mixed content)
- Multi-format document collection

Usage:
    from tests.fixtures.sample_documents import create_sample_documents

    documents = create_sample_documents(tmp_path)
    # Process with document processor, embeddings, etc.
"""

from pathlib import Path


def create_sample_documents(base_dir: Path) -> dict[str, Path]:
    """Create realistic sample documents for integration testing.

    Args:
        base_dir: Base directory for creating test documents

    Returns:
        Dictionary mapping document types to file paths
    """
    docs_dir = base_dir / "sample_documents"
    docs_dir.mkdir(exist_ok=True)

    # AI Research Paper (PDF-style content)
    research_paper = docs_dir / "ai_research_paper.txt"
    research_content = """# Advances in Large Language Models: A Comprehensive Survey

## Abstract

Large Language Models (LLMs) have revolutionized natural language processing through 
their unprecedented scale and capabilities. This paper presents a comprehensive survey 
of recent advances in LLM architectures, training methodologies, and applications.

## 1. Introduction

The field of artificial intelligence has witnessed remarkable progress with
the emergence
of large-scale language models. These models, characterized by billions of parameters,
demonstrate emergent capabilities in reasoning, code generation, and multi-modal tasks.

### 1.1 Background

Transformer architecture, introduced by Vaswani et al. (2017), laid the foundation for 
modern language models. Subsequent developments including BERT, GPT series, and recent 
models like ChatGPT have demonstrated the power of scale and pre-training.

### 1.2 Scope and Contributions

This survey covers:
- Architectural innovations in LLMs
- Training strategies and optimization techniques  
- Evaluation methodologies and benchmarks
- Applications across diverse domains
- Challenges and future research directions

## 2. Model Architectures

### 2.1 Transformer-based Architectures

The transformer architecture remains the dominant paradigm for LLMs.
Key innovations include:

- **Attention Mechanisms**: Self-attention allows models to capture
  long-range dependencies
- **Positional Encodings**: Various schemes for incorporating sequence
  position information
- **Layer Normalization**: Techniques for stabilizing training of
  deep networks

### 2.2 Scale and Parameter Efficiency

Modern LLMs achieve performance through scale, with models reaching
hundreds of billions 
of parameters. However, parameter efficiency techniques offer promising alternatives.

## 3. Training Methodologies

### 3.1 Pre-training Strategies

Effective pre-training requires careful consideration of dataset curation, training 
objectives, and optimization techniques.

### 3.2 Fine-tuning and Alignment

Post-training techniques align models with human preferences through supervised 
fine-tuning and reinforcement learning from human feedback.

## 4. Applications and Use Cases

### 4.1 Content Generation

LLMs excel at diverse generation tasks including creative writing, technical 
documentation, and code generation.

### 4.2 Knowledge-Intensive Tasks

Integration with external knowledge enables question answering systems, research 
assistance, and educational applications.

## 5. Conclusion

Large Language Models represent a paradigm shift in artificial intelligence, 
demonstrating remarkable capabilities across diverse tasks. While challenges remain, 
ongoing research promises continued progress toward more capable AI systems.
"""
    research_paper.write_text(research_content)

    # Technical Documentation (Markdown)
    tech_docs = docs_dir / "api_documentation.md"
    tech_content = """# DocMind AI API Documentation

## Overview

DocMind AI provides a comprehensive API for document processing, embedding generation, 
and intelligent retrieval. This documentation covers all available endpoints and 
integration patterns.

## Quick Start

### Installation

```bash
pip install docmind-ai
```

### Basic Usage

```python
from docmind import DocumentProcessor, QueryEngine

# Initialize processor
processor = DocumentProcessor()

# Process documents
result = await processor.process_document("path/to/document.pdf")

# Create query engine
engine = QueryEngine(documents=result.documents)

# Query documents
response = await engine.query("What are the key findings?")
```

## API Reference

### Document Processing

#### `DocumentProcessor.process_document()`

Processes a single document and extracts structured information.

**Parameters:**
- `file_path` (str): Path to the document file
- `strategy` (ProcessingStrategy): Processing strategy to use
- `options` (dict): Additional processing options

**Returns:**
- `ProcessingResult`: Structured processing result

### Embedding Generation

#### `EmbeddingEngine.create_embeddings()`

Generates embeddings for document chunks.

**Parameters:**
- `texts` (List[str]): List of text chunks
- `model` (str): Embedding model to use

**Returns:**
- `EmbeddingResult`: Generated embeddings and metadata

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCMIND_MODEL_PATH` | Path to language model | `qwen3-4b` |
| `DOCMIND_EMBEDDING_MODEL` | Embedding model name | `bge-m3` |
| `DOCMIND_VECTOR_STORE` | Vector store backend | `qdrant` |

## Advanced Features

### Multi-Agent Processing

DocMind AI uses a multi-agent architecture for complex document analysis.

### Custom Processing Pipelines

Create custom processing workflows for specific document types and requirements.
"""
    tech_docs.write_text(tech_content)

    # Business Report
    business_report = docs_dir / "quarterly_report.txt"
    business_content = """QUARTERLY BUSINESS REPORT - Q4 2024
=======================================

Executive Summary
-----------------

This report presents the performance metrics, strategic initiatives, and financial 
results for the fourth quarter of 2024. The organization demonstrated strong growth 
across key performance indicators while successfully executing on strategic objectives.

Key Highlights:
• Revenue increased 23% year-over-year to $12.5M
• Customer acquisition improved 35% with 2,847 new customers
• Product launches exceeded targets with 4 major releases
• Employee satisfaction reached 89% in annual survey

Financial Performance
--------------------

Revenue Breakdown by Segment:
- Enterprise Solutions: $7.2M (58% of total revenue)
- SMB Products: $3.1M (25% of total revenue) 
- Professional Services: $2.2M (17% of total revenue)

Profitability Metrics:
- Gross Margin: 72% (target: 70%)
- Operating Margin: 18% (target: 15%)
- EBITDA: $2.3M (improvement of $800K vs Q3)

Customer Metrics
---------------

Acquisition and Retention:
- New Customer Acquisition: 2,847 customers (+35% YoY)
- Customer Retention Rate: 94% (industry benchmark: 89%)
- Net Promoter Score: 67 (improvement from 61 in Q3)
- Average Customer Lifetime Value: $4,200

Product Development
------------------

Major Product Releases:
1. AI-Powered Analytics Suite v3.0
2. Mobile Application Redesign
3. Enterprise Security Module
4. API Platform Expansion

Research and Development:
- R&D Investment: $1.8M (14% of revenue)
- Patent Applications: 3 filed, 2 approved
- Innovation Pipeline: 12 projects in development

Strategic Initiatives
--------------------

Completed Objectives:
✓ Launched AI analytics platform ahead of schedule
✓ Expanded customer support to 24/7 availability
✓ Achieved SOC2 compliance certification
✓ Established European operations center

Q1 2025 Priorities:
• Launch machine learning automation features
• Expand into Asian markets
• Complete Series B funding round
• Implement advanced customer segmentation

Conclusion
----------

Q4 2024 demonstrated exceptional performance across all key metrics, positioning the 
organization for continued success in 2025. The focus on AI-powered solutions and 
customer satisfaction has differentiated our offerings in a competitive market.
"""
    business_report.write_text(business_content)

    # Code Documentation
    code_docs = docs_dir / "python_guide.md"
    code_content = """# Python Development Guide

## Code Style and Standards

### PEP 8 Compliance

Follow PEP 8 guidelines for Python code formatting and write clean, readable code 
with proper variable names, function organization, and documentation.

### Type Hints

Always use type hints for function parameters and return values to improve code 
clarity and enable better IDE support.

### Error Handling

Use specific exception types and provide meaningful error messages. Handle errors 
at appropriate boundaries and provide graceful degradation when possible.

## Testing Patterns

Write comprehensive tests using pytest with proper fixtures, parametrization, and 
async test support for I/O-bound operations.

## Performance Optimization

Use async/await for I/O-bound operations, implement appropriate caching strategies, 
and use generators for large datasets to manage memory efficiently.

## Documentation Standards

Write clear docstrings using Google-style format with comprehensive parameter 
descriptions, return value documentation, and usage examples.
"""
    code_docs.write_text(code_content)

    # Multi-language content for testing
    multilingual_doc = docs_dir / "multilingual_content.txt"
    multilingual_content = """Multilingual Document Processing Test

English Section:
This document contains content in multiple languages to test the document processing
pipeline's ability to handle diverse character sets and linguistic patterns.

Spanish Section:
La inteligencia artificial está revolucionando la forma en que procesamos y analizamos
documentos. Los sistemas modernos pueden manejar múltiples idiomas y formatos.

French Section:  
L'intelligence artificielle transforme notre façon de traiter l'information. Les
algorithmes d'apprentissage automatique peuvent identifier des modèles complexes.

German Section:
Künstliche Intelligenz verändert die Art, wie wir Dokumente verarbeiten und analysieren.
Moderne Sprachmodelle können komplexe Zusammenhänge verstehen.

Chinese Section:
人工智能正在改变我们处理和分析文档的方式。现代语言模型能够理解复杂的上下文关系。

Technical Terms Section:
- Machine Learning (Aprendizaje Automático, Apprentissage Automatique,
  Maschinelles Lernen, 机器学习)
- Natural Language Processing (Procesamiento de Lenguaje Natural, 自然语言处理)
- Deep Learning (Aprendizaje Profundo, Apprentissage Profond, Tiefes Lernen, 深度学习)

Special Characters and Symbols:
Mathematical: α, β, γ, δ, π, σ, ∑, ∫, ∂, √, ∞
Currency: $, €, ¥, £, ₹, ₩, ₽
Emoji: 🚀 📊 💡 🔬 🌐 📚 🎯 ✨
"""
    multilingual_doc.write_text(multilingual_content, encoding="utf-8")

    return {
        "research_paper": research_paper,
        "tech_docs": tech_docs,
        "business_report": business_report,
        "code_docs": code_docs,
        "multilingual": multilingual_doc,
    }


def create_corrupted_document(base_dir: Path) -> Path:
    """Create a document that simulates corruption for error testing."""
    docs_dir = base_dir / "sample_documents"
    docs_dir.mkdir(exist_ok=True)

    corrupted_file = docs_dir / "corrupted_document.txt"
    # Create file with unusual content that might trigger processing errors
    corrupted_content = (
        "This is a test document with problematic content\x00\x01\x02\xff"
    )
    corrupted_file.write_bytes(corrupted_content.encode("utf-8", errors="ignore"))

    return corrupted_file


def create_large_document(base_dir: Path, size_kb: int = 500) -> Path:
    """Create a large document for performance testing."""
    docs_dir = base_dir / "sample_documents"
    docs_dir.mkdir(exist_ok=True)

    large_file = docs_dir / "large_performance_test.txt"

    # Generate content to reach approximately the target size
    base_content = """Performance Testing Document

This document is designed to test the performance characteristics of the document
processing pipeline when handling large files. The content is repeated multiple
times to create a document of significant size.

Section Content:
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud 
exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Technical Information:
The document processing pipeline should efficiently handle large documents by:
- Implementing proper memory management
- Using streaming processing where appropriate
- Maintaining reasonable processing times
- Providing progress indicators for long operations

Performance Metrics:
- Processing time should scale linearly with document size
- Memory usage should remain bounded
- CPU utilization should be efficient
- I/O operations should be optimized
"""

    # Calculate repetitions needed for target size
    content_size = len(base_content.encode("utf-8"))
    repetitions = (size_kb * 1024) // content_size + 1

    full_content = base_content * repetitions
    large_file.write_text(full_content)

    return large_file
