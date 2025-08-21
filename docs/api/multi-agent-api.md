# Multi-Agent System API Documentation

## Overview

The DocMind AI Multi-Agent API provides programmatic access to the 5-agent LangGraph supervisor system. This API enables integration with external applications, automation workflows, and advanced document analysis pipelines with 128K context capability and FP8 optimization.

## Authentication & Access

### API Configuration

```python
from docmind.api import DocMindAPI
from docmind.config import APIConfig

# Initialize API with configuration
config = APIConfig(
    model_name="Qwen3-4B-Instruct-2507-FP8",
    max_context_length=131072,
    gpu_memory_utilization=0.85,
    enable_fp8_optimization=True
)

api = DocMindAPI(config)
```

### Environment Variables

```bash
# Required environment variables
export DOCMIND_MODEL_PATH="/path/to/qwen3-model"
export DOCMIND_VECTOR_STORE_PATH="/path/to/qdrant"
export DOCMIND_CACHE_DIR="/path/to/cache"

# Optional performance settings
export DOCMIND_GPU_MEMORY_UTILIZATION=0.85
export DOCMIND_ENABLE_FP8_KV_CACHE=true
export DOCMIND_MAX_PARALLEL_AGENTS=3
```

## Core API Endpoints

### Agent Management

#### Initialize Multi-Agent System

```python
POST /api/v1/agents/initialize
```

**Request Body:**
```json
{
  "config": {
    "model_name": "Qwen3-4B-Instruct-2507-FP8",
    "max_context_length": 131072,
    "agent_config": {
      "query_router": {
        "temperature": 0.1,
        "confidence_threshold": 0.8
      },
      "retrieval_expert": {
        "top_k": 20,
        "enable_reranking": true
      },
      "result_synthesizer": {
        "max_synthesis_length": 2048
      }
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "session_id": "sess_abc123def456",
  "agents_initialized": [
    "supervisor",
    "query_router", 
    "query_planner",
    "retrieval_expert",
    "result_synthesizer",
    "response_validator"
  ],
  "performance_metrics": {
    "initialization_time": 3.2,
    "vram_usage_gb": 13.5,
    "context_window_size": 131072
  }
}
```

#### Agent Health Check

```python
GET /api/v1/agents/health
```

**Response:**
```json
{
  "status": "healthy",
  "agents": {
    "supervisor": {
      "status": "active",
      "uptime": 1800.5,
      "last_activity": "2025-08-20T15:30:45Z"
    },
    "query_router": {
      "status": "active",
      "cache_hits": 156,
      "avg_response_time": 0.12
    },
    "retrieval_expert": {
      "status": "active",
      "searches_completed": 89,
      "avg_retrieval_time": 0.25
    }
  },
  "system_metrics": {
    "vram_usage_gb": 13.8,
    "gpu_utilization": 75,
    "context_utilization": 45000
  }
}
```

### Document Analysis

#### Submit Analysis Query

```python
POST /api/v1/analyze
```

**Request Body:**
```json
{
  "query": "Analyze the quarterly financial performance and identify key trends",
  "session_id": "sess_abc123def456",
  "options": {
    "max_response_tokens": 2048,
    "enable_citations": true,
    "performance_mode": "balanced",
    "context_optimization": true
  },
  "document_filters": {
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    },
    "document_types": ["pdf", "docx"],
    "tags": ["financial", "quarterly"]
  }
}
```

**Response:**
```json
{
  "request_id": "req_789xyz012",
  "session_id": "sess_abc123def456",
  "status": "success",
  "analysis_result": {
    "response": "Based on the analysis of quarterly financial documents...",
    "citations": [
      {
        "source": "Q4_2024_Financial_Report.pdf",
        "page": 15,
        "relevance_score": 0.92,
        "excerpt": "Revenue increased by 18% compared to..."
      }
    ],
    "confidence_score": 0.89,
    "agent_execution_trace": [
      {
        "agent": "query_router",
        "decision": "hybrid_search",
        "confidence": 0.85,
        "execution_time": 0.11
      },
      {
        "agent": "retrieval_expert", 
        "results_count": 15,
        "reranked": true,
        "execution_time": 0.28
      }
    ]
  },
  "performance_metrics": {
    "total_execution_time": 2.45,
    "token_usage": {
      "input_tokens": 1250,
      "output_tokens": 890,
      "total_tokens": 2140
    },
    "agent_coordination_time": 0.15,
    "parallel_execution_savings": 0.67
  }
}
```

#### Stream Analysis Response

```python
POST /api/v1/analyze/stream
```

**Request Body:** Same as above

**Streaming Response:**
```json
# Initial response
{"type": "start", "request_id": "req_789xyz012", "timestamp": "2025-08-20T15:30:45Z"}

# Agent execution updates
{"type": "agent_update", "agent": "query_router", "status": "completed", "result": {"strategy": "hybrid_search"}}
{"type": "agent_update", "agent": "retrieval_expert", "status": "in_progress", "progress": 0.6}

# Partial response chunks
{"type": "response_chunk", "content": "Based on the analysis", "chunk_id": 1}
{"type": "response_chunk", "content": " of quarterly financial", "chunk_id": 2}

# Final completion
{"type": "complete", "final_response": "...", "performance_metrics": {...}}
```

### Conversation Management

#### Create Conversation Session

```python
POST /api/v1/conversations
```

**Request Body:**
```json
{
  "name": "Financial Analysis Session",
  "description": "Q4 2024 financial document analysis",
  "options": {
    "max_conversation_length": 50,
    "context_preservation_strategy": "priority_based",
    "auto_summarization": true
  }
}
```

**Response:**
```json
{
  "session_id": "sess_abc123def456",
  "status": "created",
  "created_at": "2025-08-20T15:30:45Z",
  "configuration": {
    "max_context_tokens": 131072,
    "context_buffer": 8192,
    "agent_timeouts": {
      "query_router": 30,
      "retrieval_expert": 60,
      "result_synthesizer": 45
    }
  }
}
```

#### Get Conversation History

```python
GET /api/v1/conversations/{session_id}/history
```

**Query Parameters:**
- `limit`: Maximum number of messages (default: 50)
- `offset`: Pagination offset (default: 0)
- `include_metadata`: Include agent execution metadata (default: false)

**Response:**
```json
{
  "session_id": "sess_abc123def456",
  "messages": [
    {
      "id": "msg_001",
      "timestamp": "2025-08-20T15:30:45Z",
      "role": "user",
      "content": "Analyze the quarterly financial performance",
      "metadata": {
        "token_count": 45,
        "processing_time": 2.3
      }
    },
    {
      "id": "msg_002", 
      "timestamp": "2025-08-20T15:30:48Z",
      "role": "assistant",
      "content": "Based on the analysis of quarterly financial documents...",
      "metadata": {
        "token_count": 890,
        "agent_trace": [...],
        "citations": [...]
      }
    }
  ],
  "pagination": {
    "total_messages": 12,
    "current_page": 1,
    "total_pages": 1
  },
  "context_utilization": {
    "current_tokens": 8950,
    "max_tokens": 131072,
    "utilization_percentage": 6.8
  }
}
```

#### Clear Conversation Context

```python
DELETE /api/v1/conversations/{session_id}/context
```

**Options:**
```json
{
  "preserve_documents": true,
  "keep_last_n_messages": 3,
  "summarize_cleared_context": true
}
```

**Response:**
```json
{
  "status": "success",
  "context_cleared": {
    "messages_removed": 15,
    "tokens_freed": 45000,
    "summary_created": "Previous conversation covered financial analysis topics including..."
  },
  "new_context_size": 12500
}
```

### Document Management

#### Upload Documents

```python
POST /api/v1/documents/upload
```

**Request:** Multipart form data
- `files`: Document files (PDF, DOCX, TXT)
- `metadata`: JSON metadata for documents

**Response:**
```json
{
  "status": "success",
  "uploaded_documents": [
    {
      "document_id": "doc_abc123",
      "filename": "Q4_2024_Report.pdf",
      "size_bytes": 2045678,
      "pages": 45,
      "processing_status": "queued",
      "estimated_processing_time": 120
    }
  ],
  "processing_job_id": "job_xyz789"
}
```

#### Check Document Processing Status

```python
GET /api/v1/documents/processing/{job_id}
```

**Response:**
```json
{
  "job_id": "job_xyz789",
  "status": "completed",
  "progress": 100,
  "documents_processed": [
    {
      "document_id": "doc_abc123",
      "status": "completed",
      "chunks_created": 156,
      "embeddings_generated": 156,
      "processing_time": 89.5,
      "metadata": {
        "word_count": 12500,
        "language": "en",
        "document_type": "financial_report"
      }
    }
  ]
}
```

#### Search Documents

```python
POST /api/v1/documents/search
```

**Request Body:**
```json
{
  "query": "revenue growth quarterly comparison",
  "filters": {
    "document_types": ["pdf"],
    "date_range": {
      "start": "2024-01-01", 
      "end": "2024-12-31"
    },
    "tags": ["financial"]
  },
  "options": {
    "top_k": 20,
    "enable_reranking": true,
    "include_content": true,
    "highlight_matches": true
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "document_id": "doc_abc123",
      "filename": "Q4_2024_Report.pdf",
      "chunk_id": "chunk_456",
      "relevance_score": 0.92,
      "content": "Revenue increased by 18% compared to the previous quarter...",
      "highlighted_content": "**Revenue** increased by 18% compared to the previous **quarter**...",
      "metadata": {
        "page": 15,
        "section": "Financial Performance",
        "word_count": 250
      }
    }
  ],
  "search_metadata": {
    "total_results": 45,
    "search_time": 0.28,
    "reranking_applied": true,
    "strategy_used": "hybrid_search"
  }
}
```

## Agent-Specific Endpoints

### Query Router Agent

#### Get Routing Decision

```python
POST /api/v1/agents/query-router/route
```

**Request Body:**
```json
{
  "query": "What are the key performance indicators for Q4?",
  "context": {
    "conversation_history": [...],
    "user_preferences": {
      "preferred_search_strategy": "hybrid"
    }
  }
}
```

**Response:**
```json
{
  "routing_decision": {
    "strategy": "hybrid_search",
    "confidence": 0.87,
    "reasoning": "Query contains both specific terms (Q4) and conceptual elements (performance indicators)",
    "parameters": {
      "top_k": 15,
      "enable_reranking": true,
      "semantic_weight": 0.7,
      "keyword_weight": 0.3
    }
  },
  "execution_time": 0.11,
  "cache_hit": false
}
```

### Retrieval Expert Agent

#### Execute Retrieval

```python
POST /api/v1/agents/retrieval-expert/retrieve
```

**Request Body:**
```json
{
  "query": "quarterly revenue performance analysis",
  "routing_decision": {
    "strategy": "hybrid_search",
    "parameters": {...}
  },
  "filters": {
    "document_types": ["pdf", "xlsx"],
    "date_range": {...}
  }
}
```

**Response:**
```json
{
  "retrieval_results": [
    {
      "document_id": "doc_abc123",
      "chunk_id": "chunk_456", 
      "relevance_score": 0.92,
      "content": "Revenue analysis shows...",
      "metadata": {...}
    }
  ],
  "retrieval_metadata": {
    "strategy_executed": "hybrid_search",
    "results_count": 15,
    "reranking_applied": true,
    "execution_time": 0.28,
    "cache_utilization": 0.15
  }
}
```

### Result Synthesizer Agent

#### Synthesize Results

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