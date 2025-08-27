# DocMind AI Python API Documentation

## Overview

DocMind AI provides powerful programmatic interfaces for integrating advanced document analysis capabilities into Python applications. Built on a 5-agent LangGraph supervisor system with 128K context capability and FP8 optimization, the API enables developers to programmatically process documents, coordinate multi-agent workflows, and build custom analysis pipelines.

> **Current Architecture Note**: DocMind AI is currently implemented as a Python library with Streamlit interface. REST API endpoints are planned for future releases. This documentation covers the programmatic Python interfaces available now.

## Quick Start

```python
from src.agents.coordinator import MultiAgentCoordinator
from src.utils.document import load_documents_unstructured
from src.config import settings

# Initialize the multi-agent system
coordinator = MultiAgentCoordinator()

# Load and process documents
documents = await load_documents_unstructured(["/path/to/document.pdf"], settings)

# Process query with multi-agent coordination
response = coordinator.process_query(
    "Analyze the key insights from this document",
    context=documents
)

print(response.content)
```

## Configuration & Setup

### Environment Configuration

DocMind AI uses environment variables for configuration. Copy `.env.example` to `.env` and configure:

```bash
# Core Model Settings
DOCMIND_MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507-FP8
DOCMIND_CONTEXT_WINDOW_SIZE=131072
DOCMIND_LLM_BASE_URL=http://localhost:11434

# Multi-Agent System
DOCMIND_ENABLE_MULTI_AGENT=true
DOCMIND_AGENT_DECISION_TIMEOUT=200
DOCMIND_MAX_AGENT_RETRIES=2
DOCMIND_ENABLE_FALLBACK_RAG=true

# Performance & GPU
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_KV_CACHE_DTYPE=fp8_e5m2
VLLM_GPU_MEMORY_UTILIZATION=0.85
DOCMIND_ENABLE_GPU_ACCELERATION=true

# Document Processing
DOCMIND_CHUNK_SIZE=1500
DOCMIND_CHUNK_OVERLAP=150
DOCMIND_MAX_DOCUMENT_SIZE_MB=100

# Retrieval & Search
DOCMIND_RETRIEVAL_STRATEGY=hybrid
DOCMIND_EMBEDDING_MODEL=BAAI/bge-m3
DOCMIND_USE_RERANKING=true
```

### Python Configuration

```python
from src.config import settings

# Access configuration
print(f"Model: {settings.vllm.model}")
print(f"Context Window: {settings.vllm.context_window}")
print(f"GPU Memory: {settings.vllm.gpu_memory_utilization}")

# Modify settings programmatically
settings.agents.decision_timeout = 300
settings.processing.chunk_size = 2000
```

### Hardware Requirements

**Minimum Requirements:**

- **GPU**: RTX 4060 (16GB VRAM)
- **RAM**: 16GB system RAM
- **Storage**: 50GB free space
- **CUDA**: 12.8+ compatibility

**Recommended Setup:**

- **GPU**: RTX 4090 (16GB VRAM)
- **RAM**: 32GB system RAM
- **Storage**: 100GB NVMe SSD
- **Total VRAM Usage**: 12-14GB (Qwen3-4B-FP8 + BGE-M3 + reranker + 128K context)

## Core Python API

### Multi-Agent Coordination System

The `MultiAgentCoordinator` is the primary interface for document analysis, orchestrating 5 specialized agents using LangGraph supervisor pattern.

#### Basic Usage

```python
from src.agents.coordinator import MultiAgentCoordinator
from llama_index.core.memory import ChatMemoryBuffer

# Initialize coordinator
coordinator = MultiAgentCoordinator(
    model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
    max_context_length=131072,
    backend="vllm",
    enable_fallback=True,
    max_agent_timeout=200  # milliseconds
)

# Create memory buffer for conversation context
memory = ChatMemoryBuffer.from_defaults(token_limit=131072)

# Process a query
response = coordinator.process_query(
    query="Analyze the quarterly financial performance trends",
    context=memory
)

print(f"Response: {response.content}")
print(f"Agent Coordination Time: {response.execution_time}s")
print(f"Agents Used: {response.agents_invoked}")
```

#### Advanced Configuration

```python
# Custom agent configuration
coordinator = MultiAgentCoordinator(
    model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
    max_context_length=131072,
    backend="vllm",
    enable_fallback=True,
    max_agent_timeout=300,
    # Agent-specific configurations
    query_router_config={
        "temperature": 0.1,
        "confidence_threshold": 0.8
    },
    retrieval_config={
        "top_k": 20,
        "enable_reranking": True,
        "reranker_model": "BAAI/bge-reranker-v2-m3"
    },
    synthesis_config={
        "max_length": 2048,
        "include_citations": True
    }
)

# Process with detailed response
response = coordinator.process_query_detailed(
    query="What are the key performance indicators mentioned?",
    context=memory,
    return_agent_trace=True,
    enable_streaming=False
)

# Access detailed response information
print(f"Final Response: {response.content}")
print(f"Confidence Score: {response.confidence}")
print(f"Source Citations: {len(response.citations)}")
print(f"Agent Execution Trace:")
for step in response.agent_trace:
    print(f"  - {step.agent}: {step.action} ({step.duration}s)")
```

#### Agent System Status

```python
# Check agent system health
status = coordinator.get_system_status()

print(f"System Status: {status.overall_status}")
print(f"Active Agents: {len(status.active_agents)}")
for agent_name, agent_status in status.agents.items():
    print(f"  - {agent_name}: {agent_status.status}")
    print(f"    Uptime: {agent_status.uptime}s")
    print(f"    Last Activity: {agent_status.last_activity}")
    print(f"    Performance: {agent_status.avg_response_time}s avg")

# System resource usage
print(f"VRAM Usage: {status.vram_usage_gb:.1f}GB / {status.vram_total_gb}GB")
print(f"GPU Utilization: {status.gpu_utilization}%")
print(f"Context Utilization: {status.context_utilization} / {status.max_context}")
```

#### Async Processing

```python
import asyncio

# Async query processing for better performance
async def process_multiple_queries():
    coordinator = MultiAgentCoordinator()
    
    queries = [
        "Summarize the document's main points",
        "What are the key financial metrics?",
        "Identify potential risks mentioned"
    ]
    
    # Process queries concurrently
    tasks = [
        coordinator.aprocess_query(query, context=memory) 
        for query in queries
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"Query {i+1}: {response.content[:100]}...")
    
    return responses

# Run async processing
responses = asyncio.run(process_multiple_queries())
```

### Document Processing Pipeline

#### Document Loading and Processing

DocMind AI supports comprehensive document processing with unstructured.io integration.

```python
from src.utils.document import load_documents_unstructured
from src.config import settings

# Load documents with advanced processing
documents = await load_documents_unstructured(
    file_paths=[
        "/path/to/report.pdf",
        "/path/to/presentation.pptx",
        "/path/to/data.xlsx"
    ],
    settings=settings,
    # Processing options
    parse_images=True,
    extract_tables=True,
    enable_ocr=True,
    chunk_strategy="semantic"
)

print(f"Loaded {len(documents)} documents")
for doc in documents:
    print(f"  - {doc.metadata.get('filename')}: {len(doc.text)} chars")
```

#### Document Analysis with Filtering

```python
from src.agents.coordinator import MultiAgentCoordinator
from datetime import datetime

# Initialize coordinator and load documents
coordinator = MultiAgentCoordinator()
documents = await load_documents_unstructured(file_paths, settings)

# Analyze with filtering options
response = coordinator.analyze_documents(
    query="Analyze the quarterly financial performance and identify key trends",
    documents=documents,
    options={
        "max_response_tokens": 2048,
        "enable_citations": True,
        "performance_mode": "balanced",  # fast, balanced, thorough
        "context_optimization": True,
        "include_confidence_scores": True
    },
    # Document filtering
    filters={
        "date_range": {
            "start": datetime(2024, 1, 1),
            "end": datetime(2024, 12, 31)
        },
        "document_types": ["pdf", "docx"],
        "tags": ["financial", "quarterly"],
        "min_relevance_score": 0.7
    }
)

# Process results
print(f"Analysis Result: {response.content}")
print(f"Confidence: {response.confidence:.2f}")
print(f"Sources Used: {len(response.citations)}")

for citation in response.citations:
    print(f"  - {citation.source} (page {citation.page}): {citation.relevance:.2f}")
    print(f"    {citation.excerpt[:100]}...")
```

### Tool Factory and Custom Analysis Tools

The `ToolFactory` provides a centralized way to create analysis tools with optimal configuration.

```python
from src.agents.tool_factory import ToolFactory
from llama_index.core import VectorStoreIndex

# Create vector index from documents
index = VectorStoreIndex.from_documents(documents)

# Create tools using ToolFactory
tools = ToolFactory.create_tools_from_indexes(
    vector_index=index,
    enable_reranking=True,
    hybrid_search=True
)

print(f"Created {len(tools)} analysis tools:")
for tool in tools:
    print(f"  - {tool.metadata.name}: {tool.metadata.description}")

# Create specific tool types
vector_tool = ToolFactory.create_vector_search_tool(
    index=index,
    name="financial_search",
    description="Search financial documents for specific metrics and trends",
    similarity_top_k=10,
    enable_reranking=True
)

# Create custom query engine tool
custom_engine = index.as_query_engine(
    similarity_top_k=15,
    response_mode="tree_summarize"
)

custom_tool = ToolFactory.create_query_tool(
    query_engine=custom_engine,
    name="summarization_tool",
    description="Comprehensive document summarization with tree-based synthesis"
)
```

### Streaming Analysis

DocMind AI supports streaming analysis for real-time response generation and progress tracking.

```python
import asyncio
from typing import AsyncIterator

async def stream_analysis(query: str, documents) -> AsyncIterator[dict]:
    """Stream analysis results with progress updates."""
    coordinator = MultiAgentCoordinator()
    
    # Start streaming analysis
    async for chunk in coordinator.stream_query(
        query=query,
        documents=documents,
        include_agent_updates=True,
        chunk_size=50  # words per chunk
    ):
        yield chunk

# Example streaming usage
async def run_streaming_analysis():
    query = "Analyze quarterly performance and identify key trends"
    
    async for update in stream_analysis(query, documents):
        if update["type"] == "agent_start":
            print(f"🚀 {update['agent']} started: {update['task']}")
        
        elif update["type"] == "agent_complete":
            print(f"✅ {update['agent']} completed in {update['duration']}s")
        
        elif update["type"] == "response_chunk":
            print(update["content"], end="", flush=True)
        
        elif update["type"] == "complete":
            print(f"\n\n📊 Analysis completed:")
            print(f"  - Total time: {update['total_time']}s")
            print(f"  - Tokens used: {update['token_usage']}")
            print(f"  - Confidence: {update['confidence']:.2f}")

# Run streaming analysis
await run_streaming_analysis()
```

### Memory and Conversation Management

DocMind AI uses LlamaIndex memory buffers for maintaining conversation context and session persistence.

#### Basic Memory Management

```python
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage

# Create memory buffer with token limit
memory = ChatMemoryBuffer.from_defaults(
    token_limit=131072,  # 128K context window
    tokenizer_fn=None    # Auto-detect from model
)

# Add messages to memory
memory.put(ChatMessage(role="user", content="What are the main insights?"))
memory.put(ChatMessage(role="assistant", content="The main insights include..."))

# Get conversation history
messages = memory.get_all()
print(f"Conversation has {len(messages)} messages")
for msg in messages:
    print(f"{msg.role}: {msg.content[:50]}...")

# Check token usage
token_count = memory.get_all_token_count()
print(f"Current token usage: {token_count} / {memory.token_limit}")
```

#### Advanced Memory Configuration

```python
# Custom memory with conversation naming and metadata
class CustomChatMemory(ChatMemoryBuffer):
    def __init__(self, session_name: str, description: str = None, **kwargs):
        super().__init__(**kwargs)
        self.session_name = session_name
        self.description = description
        self.created_at = datetime.now()
        self.metadata = {}
    
    def add_metadata(self, key: str, value: any):
        self.metadata[key] = value
    
    def get_session_info(self):
        return {
            "name": self.session_name,
            "description": self.description,
            "created_at": self.created_at,
            "message_count": len(self.get_all()),
            "token_usage": self.get_all_token_count(),
            "metadata": self.metadata
        }

# Usage
memory = CustomChatMemory(
    session_name="Financial Analysis Session",
    description="Q4 2024 financial document analysis",
    token_limit=131072
)

memory.add_metadata("analysis_type", "quarterly_financial")
memory.add_metadata("document_count", len(documents))

# Process queries with persistent memory
coordinator = MultiAgentCoordinator()
response = coordinator.process_query(
    "What are the revenue trends?",
    context=memory
)

print(memory.get_session_info())
```

#### Memory Persistence

```python
import json
from pathlib import Path

# Save memory to file
def save_memory(memory: ChatMemoryBuffer, filepath: str):
    """Save conversation memory to JSON file."""
    messages = memory.get_all()
    session_data = {
        "session_info": memory.get_session_info() if hasattr(memory, 'get_session_info') else {},
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "additional_kwargs": getattr(msg, 'additional_kwargs', {})
            }
            for msg in messages
        ],
        "token_limit": memory.token_limit,
        "saved_at": datetime.now().isoformat()
    }
    
    Path(filepath).write_text(json.dumps(session_data, indent=2))
    print(f"Memory saved to {filepath}")

# Load memory from file
def load_memory(filepath: str) -> ChatMemoryBuffer:
    """Load conversation memory from JSON file."""
    session_data = json.loads(Path(filepath).read_text())
    
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=session_data.get("token_limit", 131072)
    )
    
    # Restore messages
    for msg_data in session_data.get("messages", []):
        memory.put(ChatMessage(
            role=msg_data["role"],
            content=msg_data["content"],
            additional_kwargs=msg_data.get("additional_kwargs", {})
        ))
    
    print(f"Memory loaded from {filepath}")
    return memory

# Usage
save_memory(memory, "session_financial_analysis.json")
restored_memory = load_memory("session_financial_analysis.json")
```

### Document Search and Retrieval

DocMind AI provides advanced search capabilities with hybrid retrieval, reranking, and semantic filtering.

```python
from src.utils.embedding import create_index_async
from src.retrieval import (
    create_adaptive_router_engine,
    configure_router_settings
)

# Create searchable index from documents
index = await create_index_async(
    documents=documents,
    settings=settings,
    # Advanced indexing options
    embedding_model="BAAI/bge-m3",
    enable_sparse_embeddings=True,
    chunk_size=1500,
    chunk_overlap=150
)

# Create adaptive router engine for intelligent search
router_engine = create_adaptive_router_engine(
    vector_index=index,
    enable_hybrid_search=True,
    reranker_model="BAAI/bge-reranker-v2-m3",
    fusion_method="rrf",  # Reciprocal Rank Fusion
    fusion_alpha=0.7
)

configure_router_settings(router_engine)

# Perform searches
response = router_engine.query(
    "What are the key financial metrics mentioned?",
    # Search configuration
    similarity_top_k=20,
    rerank_top_k=5,
    include_metadata=True
)

print(f"Response: {response.response}")
print(f"Source Nodes: {len(response.source_nodes)}")
for node in response.source_nodes:
    print(f"  - Score: {node.score:.3f}")
    print(f"    Source: {node.metadata.get('filename', 'Unknown')}")
    print(f"    Content: {node.text[:100]}...")
```

### Advanced Search with Filtering

```python
from datetime import datetime
from typing import List, Dict, Any

def search_documents_with_filters(
    query: str,
    index: Any,
    filters: Dict[str, Any] = None,
    search_options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Search documents with comprehensive filtering options."""
    
    # Default search options
    default_options = {
        "similarity_top_k": 10,
        "enable_reranking": True,
        "rerank_top_k": 5,
        "include_content": True,
        "highlight_matches": True,
        "min_relevance_score": 0.7
    }
    search_options = {**default_options, **(search_options or {})}
    
    # Apply metadata filters
    metadata_filters = []
    if filters:
        if "document_types" in filters:
            metadata_filters.append(
                MetadataFilter(key="file_type", value=filters["document_types"], operator=FilterOperator.IN)
            )
        
        if "date_range" in filters:
            date_range = filters["date_range"]
            metadata_filters.extend([
                MetadataFilter(key="created_date", value=date_range["start"], operator=FilterOperator.GTE),
                MetadataFilter(key="created_date", value=date_range["end"], operator=FilterOperator.LTE)
            ])
        
        if "tags" in filters:
            for tag in filters["tags"]:
                metadata_filters.append(
                    MetadataFilter(key="tags", value=tag, operator=FilterOperator.CONTAINS)
                )
    
    # Create filtered retriever
    retriever = index.as_retriever(
        similarity_top_k=search_options["similarity_top_k"],
        filters=MetadataFilters(filters=metadata_filters) if metadata_filters else None
    )
    
    # Add reranking if enabled
    if search_options["enable_reranking"]:
        from llama_index.postprocessor.colbert_rerank import ColbertRerank
        reranker = ColbertRerank(
            top_n=search_options["rerank_top_k"],
            model="BAAI/bge-reranker-v2-m3"
        )
        retriever = retriever.with_node_postprocessors([reranker])
    
    # Execute search
    nodes = retriever.retrieve(query)
    
    # Filter by relevance score
    filtered_nodes = [
        node for node in nodes 
        if node.score >= search_options["min_relevance_score"]
    ]
    
    # Format results
    results = {
        "query": query,
        "total_results": len(filtered_nodes),
        "results": [],
        "search_metadata": {
            "search_time": 0.28,  # Would be measured in real implementation
            "reranking_applied": search_options["enable_reranking"],
            "filters_applied": len(metadata_filters),
            "strategy_used": "hybrid_search"
        }
    }
    
    for node in filtered_nodes:
        result = {
            "document_id": node.metadata.get("document_id"),
            "filename": node.metadata.get("filename"),
            "chunk_id": node.id_,
            "relevance_score": node.score,
            "metadata": {
                "page": node.metadata.get("page"),
                "section": node.metadata.get("section"),
                "word_count": len(node.text.split())
            }
        }
        
        if search_options["include_content"]:
            result["content"] = node.text
            
            if search_options["highlight_matches"]:
                # Simple highlighting (in production, use more sophisticated methods)
                highlighted = node.text
                for term in query.lower().split():
                    highlighted = highlighted.replace(
                        term, f"**{term}**", 1  # Highlight first occurrence
                    )
                result["highlighted_content"] = highlighted
        
        results["results"].append(result)
    
    return results

# Usage example
filters = {
    "document_types": ["pdf", "docx"],
    "date_range": {
        "start": datetime(2024, 1, 1),
        "end": datetime(2024, 12, 31)
    },
    "tags": ["financial", "quarterly"]
}

search_results = search_documents_with_filters(
    query="revenue growth quarterly comparison",
    index=index,
    filters=filters,
    search_options={
        "similarity_top_k": 20,
        "enable_reranking": True,
        "rerank_top_k": 5,
        "min_relevance_score": 0.8
    }
)

print(f"Found {search_results['total_results']} relevant results")
for result in search_results["results"][:3]:  # Show top 3
    print(f"📄 {result['filename']} (Score: {result['relevance_score']:.3f})")
    print(f"   {result['content'][:150]}...")
```

### Individual Agent Interfaces

DocMind AI provides direct access to individual agents for specialized processing tasks.

#### Query Router Agent

The Router Agent analyzes queries and determines optimal processing strategies.

```python
from src.agents.tools import route_query

# Direct router usage
routing_decision = await route_query(
    query="What are the key performance indicators for Q4?",
    context={
        "conversation_history": memory.get_all()[-5:],  # Last 5 messages
        "document_metadata": {
            "total_documents": len(documents),
            "document_types": ["pdf", "xlsx", "docx"],
            "date_range": "Q4_2024"
        },
        "user_preferences": {
            "preferred_search_strategy": "hybrid",
            "detail_level": "comprehensive"
        }
    }
)

print(f"Routing Decision: {routing_decision}")
print(f"  Strategy: {routing_decision['strategy']}")
print(f"  Confidence: {routing_decision['confidence']:.2f}")
print(f"  Reasoning: {routing_decision['reasoning']}")
print(f"  Parameters: {routing_decision['parameters']}")
print(f"  Execution Time: {routing_decision['execution_time']}s")
```

#### Retrieval Expert Agent

```python
from src.agents.tools import retrieve_documents

# Execute specialized retrieval
retrieval_results = await retrieve_documents(
    query="quarterly revenue performance analysis",
    routing_decision=routing_decision,
    filters={
        "document_types": ["pdf", "xlsx"],
        "relevance_threshold": 0.8,
        "max_results": 15
    }
)

print(f"Retrieval Results:")
print(f"  Strategy Executed: {retrieval_results['strategy_executed']}")
print(f"  Results Count: {retrieval_results['results_count']}")
print(f"  Reranking Applied: {retrieval_results['reranking_applied']}")
print(f"  Execution Time: {retrieval_results['execution_time']}s")
print(f"  Cache Utilization: {retrieval_results['cache_utilization']:.2f}")

# Access individual results
for i, result in enumerate(retrieval_results['results'][:3]):
    print(f"\n  Result {i+1}:")
    print(f"    Document: {result['document_id']}")
    print(f"    Relevance: {result['relevance_score']:.3f}")
    print(f"    Content: {result['content'][:100]}...")
```

#### Planning Agent

```python
from src.agents.tools import plan_query

# Complex query planning
query_plan = await plan_query(
    query="Compare Q3 vs Q4 performance across all business units and identify improvement opportunities",
    context={
        "available_documents": documents,
        "analysis_depth": "comprehensive",
        "comparison_timeframes": ["Q3_2024", "Q4_2024"]
    }
)

print(f"Query Plan:")
print(f"  Complexity Level: {query_plan['complexity_level']}")
print(f"  Sub-tasks: {len(query_plan['sub_tasks'])}")
for i, task in enumerate(query_plan['sub_tasks']):
    print(f"    {i+1}. {task['description']}")
    print(f"       Agent: {task['assigned_agent']}")
    print(f"       Priority: {task['priority']}")
    print(f"       Est. Time: {task['estimated_time']}s")

print(f"  Execution Strategy: {query_plan['execution_strategy']}")
print(f"  Resource Requirements: {query_plan['resource_requirements']}")
```

#### Synthesis Agent

```python
from src.agents.tools import synthesize_results

# Synthesize multiple results
synthesis_result = await synthesize_results(
    query="Analyze quarterly performance trends",
    retrieval_results=retrieval_results['results'],
    synthesis_options={
        "max_length": 2048,
        "include_citations": True,
        "confidence_threshold": 0.8,
        "format": "analytical_report",
        "remove_duplicates": True,
        "fact_checking": True
    }
)

print(f"Synthesis Results:")
print(f"  Content: {synthesis_result['synthesized_response']['content']}")
print(f"  Confidence: {synthesis_result['synthesized_response']['confidence_score']:.2f}")
print(f"  Key Insights: {len(synthesis_result['synthesized_response']['key_insights'])}")
for insight in synthesis_result['synthesized_response']['key_insights']:
    print(f"    • {insight}")

print(f"\n  Citations: {len(synthesis_result['synthesized_response']['citations'])}")
for citation in synthesis_result['synthesized_response']['citations']:
    print(f"    - {citation['source']} (confidence: {citation['confidence']:.2f})")

print(f"\n  Synthesis Metadata:")
print(f"    Sources Used: {synthesis_result['synthesis_metadata']['sources_used']}")
print(f"    Conflicts Resolved: {synthesis_result['synthesis_metadata']['conflicts_resolved']}")
print(f"    Execution Time: {synthesis_result['synthesis_metadata']['execution_time']}s")
```

#### Validation Agent

```python
from src.agents.tools import validate_response

# Validate analysis results
validation_result = await validate_response(
    original_query="Analyze quarterly performance trends",
    generated_response=synthesis_result['synthesized_response']['content'],
    source_documents=retrieval_results['results'],
    validation_criteria={
        "factual_accuracy": True,
        "completeness": True,
        "relevance": True,
        "consistency": True,
        "citation_verification": True
    }
)

print(f"Validation Results:")
print(f"  Overall Score: {validation_result['validation_score']:.2f}")
print(f"  Passed Validation: {validation_result['passed']}")

print(f"\n  Detailed Scores:")
for criterion, score in validation_result['detailed_scores'].items():
    status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
    print(f"    {criterion}: {score:.2f} {status}")

if validation_result['issues_found']:
    print(f"\n  Issues Found: {len(validation_result['issues_found'])}")
    for issue in validation_result['issues_found']:
        print(f"    • {issue['type']}: {issue['description']}")
        print(f"      Severity: {issue['severity']}")
        print(f"      Suggestion: {issue['suggestion']}")

print(f"\n  Validation Metadata:")
print(f"    Sources Verified: {validation_result['validation_metadata']['sources_verified']}")
print(f"    Cross-references Checked: {validation_result['validation_metadata']['cross_references']}")
print(f"    Validation Time: {validation_result['validation_metadata']['execution_time']}s")
```

### Performance Monitoring and Metrics

DocMind AI provides comprehensive performance monitoring for system optimization and troubleshooting.

#### System Performance Metrics

```python
from src.utils.monitoring import PerformanceMonitor
from src.core.infrastructure import get_hardware_status

# Initialize performance monitoring
monitor = PerformanceMonitor()

# Get real-time system metrics
metrics = monitor.get_realtime_metrics()

print(f"System Performance Report:")
print(f"\n🚀 Model Performance:")
print(f"  - Average Decode Speed: {metrics['model_performance']['avg_decode_tps']:.1f} tok/s")
print(f"  - Average Prefill Speed: {metrics['model_performance']['avg_prefill_tps']:.1f} tok/s")
print(f"  - Context Utilization: {metrics['model_performance']['context_utilization']:.1%}")

print(f"\n💾 GPU Status:")
print(f"  - GPU Utilization: {metrics['gpu']['utilization_percent']}%")
print(f"  - VRAM Used: {metrics['gpu']['memory_used_gb']:.1f}GB / {metrics['gpu']['memory_total_gb']}GB")
print(f"  - GPU Temperature: {metrics['gpu']['temperature_celsius']}°C")
print(f"  - Power Draw: {metrics['gpu']['power_draw_watts']}W")

print(f"\n🤖 Agent Performance:")
print(f"  - Avg Coordination Time: {metrics['agent_performance']['avg_coordination_time']:.3f}s")
print(f"  - Parallel Execution Efficiency: {metrics['agent_performance']['parallel_execution_efficiency']:.1%}")
print(f"  - Cache Hit Rate: {metrics['agent_performance']['cache_hit_rate']:.1%}")
```

#### Performance History and Analysis

```python
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Get performance history
history = monitor.get_performance_history(
    start_time=datetime.now() - timedelta(hours=1),
    end_time=datetime.now(),
    interval="5m",
    metrics=["decode_tps", "vram_usage_gb", "agent_coordination_time"]
)

print(f"Performance History ({len(history['decode_tps'])} data points):")
for metric_name, data_points in history.items():
    recent_avg = sum(point["value"] for point in data_points[-5:]) / 5
    print(f"  - {metric_name}: {recent_avg:.2f} (5min avg)")

# Plot performance trends
def plot_performance_trends(history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DocMind AI Performance Trends")
    
    # Decode speed trend
    timestamps = [point["timestamp"] for point in history["decode_tps"]]
    decode_speeds = [point["value"] for point in history["decode_tps"]]
    
    axes[0,0].plot(timestamps, decode_speeds)
    axes[0,0].set_title("Decode Speed (tokens/sec)")
    axes[0,0].set_ylabel("Tokens/sec")
    
    # VRAM usage trend
    vram_usage = [point["value"] for point in history["vram_usage_gb"]]
    axes[0,1].plot(timestamps, vram_usage)
    axes[0,1].set_title("VRAM Usage (GB)")
    axes[0,1].set_ylabel("GB")
    
    # Agent coordination time
    coord_times = [point["value"] for point in history["agent_coordination_time"]]
    axes[1,0].plot(timestamps, coord_times)
    axes[1,0].set_title("Agent Coordination Time (sec)")
    axes[1,0].set_ylabel("Seconds")
    
    plt.tight_layout()
    plt.savefig("performance_trends.png")
    print("Performance trends saved to performance_trends.png")

# Generate performance report
plot_performance_trends(history)
```

### Configuration Management

DocMind AI provides flexible configuration management through the unified settings system.

#### Dynamic Configuration Updates

```python
from src.config import settings
from src.config.settings import DocMindSettings

# View current configuration
print(f"Current Configuration:")
print(f"  Model: {settings.vllm.model}")
print(f"  Context Window: {settings.vllm.context_window}")
print(f"  GPU Memory: {settings.vllm.gpu_memory_utilization}")
print(f"  Agent Timeout: {settings.agents.decision_timeout}ms")
print(f"  Retrieval Strategy: {settings.retrieval.strategy}")

# Temporarily modify settings for specific operations
with settings.temporary_override({
    "agents.decision_timeout": 500,
    "retrieval.top_k": 15,
    "vllm.max_tokens": 4096
}):
    # Operations with modified settings
    response = coordinator.process_query(
        "Complex analysis requiring more time and tokens",
        context=memory
    )

print(f"Settings restored: {settings.agents.decision_timeout}ms")
```

#### Configuration Validation and Optimization

```python
# Validate current configuration for your hardware
validation_result = settings.validate_configuration()

if validation_result.is_valid:
    print("✅ Configuration is valid and optimized")
else:
    print("⚠️ Configuration issues found:")
    for issue in validation_result.issues:
        print(f"  - {issue.severity}: {issue.message}")
        print(f"    Suggestion: {issue.suggestion}")

# Get hardware-optimized configuration
hardware_config = settings.optimize_for_hardware()
print(f"\n🚀 Hardware-Optimized Settings:")
print(f"  - Suggested Model: {hardware_config.model}")
print(f"  - Optimal GPU Memory: {hardware_config.gpu_memory_utilization}")
print(f"  - Recommended Batch Size: {hardware_config.batch_size}")
print(f"  - Context Window: {hardware_config.context_window}")

# Apply optimized configuration
if input("Apply optimized settings? (y/n): ").lower() == 'y':
    settings.apply_hardware_optimization(hardware_config)
    print("✅ Settings optimized for your hardware")
```

### Error Handling and Troubleshooting

Comprehensive error handling patterns and troubleshooting utilities.

#### Common Error Scenarios

```python
from src.utils.exceptions import (
    DocMindError, 
    AgentCoordinationError, 
    ModelLoadError, 
    InsufficientVRAMError,
    DocumentProcessingError
)
from src.utils.troubleshooting import TroubleshootingAssistant

troubleshooter = TroubleshootingAssistant()

try:
    coordinator = MultiAgentCoordinator()
    response = coordinator.process_query("Test query", context=memory)
    
except InsufficientVRAMError as e:
    print(f"🚀 VRAM Issue: {e}")
    # Get specific troubleshooting advice
    advice = troubleshooter.diagnose_vram_issue(e)
    print(f"Recommendations:")
    for rec in advice.recommendations:
        print(f"  - {rec}")
    
    # Attempt automatic remediation
    if troubleshooter.can_auto_fix(e):
        print("Attempting automatic fix...")
        success = troubleshooter.auto_fix(e)
        if success:
            print("✅ Issue resolved automatically")
            # Retry operation
            response = coordinator.process_query("Test query", context=memory)

except AgentCoordinationError as e:
    print(f"🤖 Agent Coordination Issue: {e}")
    
    # Diagnostic information
    diagnostics = troubleshooter.diagnose_agent_issue(e)
    print(f"Diagnostics:")
    print(f"  - Failed Agent: {diagnostics.failed_agent}")
    print(f"  - Error Stage: {diagnostics.error_stage}")
    print(f"  - System State: {diagnostics.system_state}")
    
    # Suggested actions
    for action in diagnostics.suggested_actions:
        print(f"  → {action}")

except DocumentProcessingError as e:
    print(f"📄 Document Processing Issue: {e}")
    
    # File-specific diagnostics
    file_diagnostics = troubleshooter.diagnose_document_issue(e)
    print(f"File Analysis:")
    print(f"  - File Type: {file_diagnostics.file_type}")
    print(f"  - File Size: {file_diagnostics.file_size_mb:.1f}MB")
    print(f"  - Issues Found: {len(file_diagnostics.issues)}")
    
    for issue in file_diagnostics.issues:
        print(f"    ⚠️ {issue.type}: {issue.description}")
        if issue.fix_available:
            print(f"      Fix: {issue.suggested_fix}")

except ModelLoadError as e:
    print(f"🤖 Model Loading Issue: {e}")
    
    # Model diagnostics
    model_diagnostics = troubleshooter.diagnose_model_issue(e)
    print(f"Model Status:")
    print(f"  - Model Path: {model_diagnostics.model_path}")
    print(f"  - Available: {model_diagnostics.model_available}")
    print(f"  - Compatible: {model_diagnostics.hardware_compatible}")
    print(f"  - Backend Status: {model_diagnostics.backend_status}")
    
    if not model_diagnostics.model_available:
        print(f"\n📦 Auto-download available models:")
        for model in model_diagnostics.available_models:
            print(f"  - {model.name} ({model.size_gb:.1f}GB)")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    
    # General system diagnostics
    system_status = troubleshooter.get_system_diagnostics()
    print(f"\n🔧 System Diagnostics:")
    print(f"  - GPU Available: {system_status.gpu_available}")
    print(f"  - VRAM Free: {system_status.vram_free_gb:.1f}GB")
    print(f"  - Ollama Status: {system_status.ollama_status}")
    print(f"  - Model Status: {system_status.model_status}")
    print(f"  - Agent System: {system_status.agent_system_status}")
    
    # Create support report
    support_report = troubleshooter.generate_support_report(e)
    support_report_path = "docmind_support_report.json"
    with open(support_report_path, "w") as f:
        json.dump(support_report, f, indent=2)
    print(f"\n📊 Support report saved to: {support_report_path}")
```

### Integration Examples and Best Practices

#### Integration Patterns

```python
# Enterprise Integration Pattern
class DocMindIntegration:
    """Production-ready DocMind AI integration class."""
    
    def __init__(self, config_path: str = None):
        self.settings = DocMindSettings(_env_file=config_path)
        self.coordinator = None
        self.memory_store = {}
        self.performance_monitor = PerformanceMonitor()
    
    async def initialize(self):
        """Initialize the DocMind system."""
        try:
            self.coordinator = MultiAgentCoordinator(
                model_path=self.settings.vllm.model,
                max_context_length=self.settings.vllm.context_window,
                backend="vllm",
                enable_fallback=True
            )
            logger.info("DocMind AI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DocMind AI: {e}")
            raise
    
    async def process_documents(
        self, 
        file_paths: List[str], 
        session_id: str = None
    ) -> Dict[str, Any]:
        """Process documents and create searchable index."""
        
        session_id = session_id or f"session_{int(time.time())}"
        
        # Load documents
        documents = await load_documents_unstructured(file_paths, self.settings)
        
        # Create index
        index = await create_index_async(documents, self.settings)
        
        # Store session data
        self.memory_store[session_id] = {
            "documents": documents,
            "index": index,
            "memory": ChatMemoryBuffer.from_defaults(
                token_limit=self.settings.vllm.context_window
            ),
            "created_at": datetime.now(),
            "document_count": len(documents)
        }
        
        return {
            "session_id": session_id,
            "status": "ready",
            "document_count": len(documents),
            "processing_time": "measured_in_production"
        }
    
    async def analyze_query(
        self, 
        query: str, 
        session_id: str,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze query against processed documents."""
        
        if session_id not in self.memory_store:
            raise ValueError(f"Session {session_id} not found")
        
        session_data = self.memory_store[session_id]
        
        # Process query
        response = await self.coordinator.aprocess_query(
            query=query,
            context=session_data["memory"],
            documents=session_data["documents"],
            **(options or {})
        )
        
        # Update memory
        session_data["memory"].put(
            ChatMessage(role="user", content=query)
        )
        session_data["memory"].put(
            ChatMessage(role="assistant", content=response.content)
        )
        
        return {
            "response": response.content,
            "confidence": response.confidence,
            "citations": response.citations,
            "agent_trace": response.agent_trace,
            "performance_metrics": response.performance_metrics
        }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status for a specific session."""
        if session_id not in self.memory_store:
            return {"status": "not_found"}
        
        session_data = self.memory_store[session_id]
        return {
            "status": "active",
            "created_at": session_data["created_at"].isoformat(),
            "document_count": session_data["document_count"],
            "message_count": len(session_data["memory"].get_all()),
            "token_usage": session_data["memory"].get_all_token_count(),
            "context_utilization": session_data["memory"].get_all_token_count() / self.settings.vllm.context_window
        }
    
    def cleanup_session(self, session_id: str):
        """Clean up session data."""
        if session_id in self.memory_store:
            del self.memory_store[session_id]
            logger.info(f"Session {session_id} cleaned up")

# Usage example
integration = DocMindIntegration(config_path=".env")
await integration.initialize()

# Process documents
result = await integration.process_documents(
    file_paths=["/path/to/report.pdf"],
    session_id="financial_analysis_2024"
)

# Analyze queries
analysis = await integration.analyze_query(
    query="What are the key financial trends?",
    session_id="financial_analysis_2024",
    options={"enable_citations": True, "confidence_threshold": 0.8}
)

print(f"Analysis: {analysis['response']}")
print(f"Confidence: {analysis['confidence']:.2f}")
```

### Future Development Roadmap

#### Planned REST API (Coming Soon)

While DocMind AI currently provides rich Python APIs, we're developing a comprehensive REST API for broader integration support.

**Planned REST Endpoints:**

```http
# Document Management
POST   /api/v1/documents/upload
GET    /api/v1/documents/{document_id}
DELETE /api/v1/documents/{document_id}
POST   /api/v1/documents/search

# Analysis and Query Processing  
POST   /api/v1/analyze
POST   /api/v1/analyze/stream
GET    /api/v1/analyze/{request_id}/status

# Session Management
POST   /api/v1/sessions
GET    /api/v1/sessions/{session_id}
DELETE /api/v1/sessions/{session_id}
GET    /api/v1/sessions/{session_id}/history

# Agent System
POST   /api/v1/agents/coordinate
GET    /api/v1/agents/status
POST   /api/v1/agents/{agent_id}/execute

# System Monitoring
GET    /api/v1/metrics/system
GET    /api/v1/metrics/performance
GET    /api/v1/health
```

**Migration Path:**

Current Python API users will have a clear migration path:

```python
# Current Python API
coordinator = MultiAgentCoordinator()
response = coordinator.process_query(query, context)

# Future REST API (similar interface)
import httpx

response = httpx.post(
    "http://localhost:8000/api/v1/analyze",
    json={"query": query, "session_id": session_id}
)
```

#### Upcoming Features

**Q2 2025:**

- REST API Beta Release
- WebSocket streaming support
- Enhanced GraphRAG integration
- Advanced DSPy optimization

**Q3 2025:**

- Multi-tenant support
- Enterprise authentication
- Batch processing API
- Advanced monitoring dashboard

**Q4 2025:**

- Cloud deployment options
- Kubernetes operators
- Advanced security features
- Multi-modal analysis (images, audio)

#### Community and Contributions

**Getting Involved:**

```bash
# Contribute to development
git clone https://github.com/BjornMelin/docmind-ai-llm
cd docmind-ai-llm
uv sync --extra dev

# Run development tests
uv run python scripts/run_tests.py --unit --integration

# Submit feature requests
# Open GitHub issues with [FEATURE] prefix
```

**Documentation Contributions:**

- API documentation improvements
- Integration examples
- Performance optimization guides
- Troubleshooting solutions

---

## Summary

DocMind AI provides comprehensive Python APIs for document analysis through:

- **Multi-Agent Coordination**: 5-agent system with LangGraph supervisor
- **Advanced Document Processing**: Unstructured.io integration with multimodal support
- **Hybrid Search**: BGE-M3 embeddings with reranking and RRF fusion
- **Performance Optimization**: FP8 quantization, FlashInfer backend, 128K context
- **Flexible Configuration**: Pydantic-based settings with hardware optimization
- **Enterprise Features**: Memory management, session persistence, error handling

**Getting Started:**

1. Install dependencies: `uv sync --extra gpu`
2. Configure environment: Copy `.env.example` to `.env`
3. Initialize system: Follow the Quick Start guide
4. Explore examples: Check integration patterns
5. Monitor performance: Use built-in metrics and troubleshooting

For production deployments, use the enterprise integration patterns and follow the performance optimization guidelines.

**Support and Community:**

- **Documentation**: [Developer Guides](../developers/)
- **Issues**: GitHub repository  
- **Performance**: [GPU Validation Scripts](../developers/gpu-and-performance.md)
- **Advanced Features**: [Configuration Guide](../user/advanced-features.md)

The REST API is under active development and will provide the same powerful capabilities through HTTP endpoints, making DocMind AI accessible to any programming language or platform.

### Synthesize Results

```python
POST /api/v1/agents/result-synthesizer/synthesize
```

**Request Body:**

```json
{
  "query": "Analyze quarterly performance trends",
  "retrieval_results": [...],
  "synthesis_options": {
    "max_length": 2048,
    "include_citations": true,
    "confidence_threshold": 0.8,
    "format": "analytical_report"
  }
}
```

**Response:**

```json
{
  "synthesized_response": {
    "content": "Based on the comprehensive analysis of quarterly data...",
    "citations": [...],
    "confidence_score": 0.89,
    "key_insights": [
      "Revenue growth of 18% quarter-over-quarter",
      "Improved profit margins in Q4",
      "Strong performance in international markets"
    ]
  },
  "synthesis_metadata": {
    "sources_used": 12,
    "conflicts_resolved": 2,
    "execution_time": 0.45
  }
}
```

## Performance Monitoring API

### System Metrics

#### Get Real-time Metrics

```python
GET /api/v1/metrics/realtime
```

**Response:**

```json
{
  "timestamp": "2025-08-20T15:30:45Z",
  "system_metrics": {
    "gpu": {
      "utilization_percent": 78,
      "memory_used_gb": 13.8,
      "memory_total_gb": 16.0,
      "temperature_celsius": 72,
      "power_draw_watts": 285
    },
    "model_performance": {
      "avg_decode_tps": 145.2,
      "avg_prefill_tps": 1150.8,
      "context_utilization": 0.35
    },
    "agent_performance": {
      "avg_coordination_time": 0.15,
      "parallel_execution_efficiency": 0.73,
      "cache_hit_rate": 0.42
    }
  }
}
```

#### Get Performance History

```python
GET /api/v1/metrics/history
```

**Query Parameters:**

- `start_time`: ISO timestamp
- `end_time`: ISO timestamp  
- `interval`: aggregation interval (1m, 5m, 1h)
- `metrics`: comma-separated metric names

**Response:**

```json
{
  "time_range": {
    "start": "2025-08-20T14:30:45Z",
    "end": "2025-08-20T15:30:45Z",
    "interval": "5m"
  },
  "metrics": {
    "decode_tps": [
      {"timestamp": "2025-08-20T14:30:00Z", "value": 142.1},
      {"timestamp": "2025-08-20T14:35:00Z", "value": 145.8}
    ],
    "vram_usage_gb": [
      {"timestamp": "2025-08-20T14:30:00Z", "value": 13.5},
      {"timestamp": "2025-08-20T14:35:00Z", "value": 13.7}
    ]
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "AGENT_COORDINATION_FAILED",
    "message": "Query router agent failed to process request",
    "details": {
      "agent": "query_router",
      "error_type": "timeout",
      "timeout_duration": 30,
      "retry_count": 2
    },
    "request_id": "req_789xyz012",
    "timestamp": "2025-08-20T15:30:45Z"
  }
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_SESSION` | Session ID not found or expired | 404 |
| `CONTEXT_OVERFLOW` | Request exceeds context window | 413 |
| `AGENT_TIMEOUT` | Agent failed to respond within timeout | 408 |
| `INSUFFICIENT_VRAM` | GPU memory insufficient for request | 503 |
| `MODEL_NOT_LOADED` | Qwen3 model not properly initialized | 503 |
| `DOCUMENT_NOT_FOUND` | Referenced document not in vector store | 404 |
| `RATE_LIMIT_EXCEEDED` | Too many requests per time window | 429 |

## Rate Limiting

### Request Limits

| Endpoint | Limit | Window |
|----------|--------|---------|
| `/analyze` | 60 requests | 1 hour |
| `/analyze/stream` | 30 requests | 1 hour |
| `/documents/upload` | 10 requests | 1 hour |
| `/documents/search` | 300 requests | 1 hour |
| `/metrics/*` | 1000 requests | 1 hour |

### Rate Limit Headers

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1692537845
X-RateLimit-Window: 3600
```

## SDK Examples

### Python SDK

```python
from docmind_sdk import DocMindClient

# Initialize client
client = DocMindClient(
    base_url="http://localhost:8000",
    session_id="sess_abc123def456"
)

# Simple analysis
result = await client.analyze(
    query="Summarize the key financial trends",
    options={"enable_citations": True}
)

# Streaming analysis
async for chunk in client.analyze_stream(
    query="Detailed market analysis with recommendations"
):
    print(chunk.content, end="")

# Document management
upload_result = await client.upload_documents([
    "/path/to/report.pdf",
    "/path/to/presentation.pptx"
])

# Search documents
search_results = await client.search_documents(
    query="revenue projections 2025",
    filters={"document_types": ["pdf"]}
)
```

### JavaScript/TypeScript SDK

```javascript
import { DocMindClient } from '@docmind/sdk';

const client = new DocMindClient({
  baseURL: 'http://localhost:8000',
  sessionId: 'sess_abc123def456'
});

// Analysis with error handling
try {
  const result = await client.analyze({
    query: 'Analyze competitive landscape',
    options: { performanceMode: 'balanced' }
  });
  
  console.log(result.analysisResult.response);
  console.log(result.performanceMetrics);
  
} catch (error) {
  if (error.code === 'CONTEXT_OVERFLOW') {
    // Handle context overflow
    console.log('Query too long, trying with shorter context');
  }
}

// Real-time metrics
const metrics = await client.getRealtimeMetrics();
console.log(`GPU Usage: ${metrics.systemMetrics.gpu.utilizationPercent}%`);
```

## Best Practices

### Optimal Usage Patterns

1. **Session Management**: Reuse sessions for related queries to benefit from context preservation
2. **Performance Mode Selection**: Use "fast" for simple queries, "balanced" for most cases, "thorough" for complex analysis
3. **Context Optimization**: Enable context optimization for long conversations
4. **Caching**: Structure similar queries to benefit from agent caching
5. **Error Handling**: Always implement retry logic with exponential backoff

### Performance Optimization

1. **Batch Operations**: Group multiple document uploads and searches
2. **Streaming**: Use streaming endpoints for long responses to improve perceived performance
3. **Filter Usage**: Apply document filters to reduce retrieval scope
4. **Context Management**: Regularly clear conversation context for long sessions

For additional API integration examples, see the [SDK documentation](https://github.com/BjornMelin/docmind-ai-sdk) and [integration examples](../examples/).
