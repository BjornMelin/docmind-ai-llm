# DocMind AI REST API Documentation

## Overview

DocMind AI REST API provides HTTP endpoints for integrating advanced document analysis capabilities into any application. The API is built on the same 5-agent LangGraph supervisor system that powers the Python library, offering full functionality through RESTful interfaces.

> **Development Status**: The REST API is currently under active development. This documentation describes the planned API structure and endpoints. For current functionality, see the [Internal Python API](./internal-api.md).

## Authentication

### API Key Authentication

```http
Authorization: Bearer your-api-key-here
Content-Type: application/json
```

### Session-based Authentication

```http
X-Session-ID: sess_abc123def456
Content-Type: application/json
```

## Base URL

```
http://localhost:8000/api/v1
```

## Document Management

### Upload Documents

```http
POST /api/v1/documents/upload
```

**Request Body:**

```json
{
  "files": [
    {
      "name": "quarterly_report.pdf",
      "content": "base64_encoded_content",
      "content_type": "application/pdf"
    }
  ],
  "session_id": "sess_abc123def456",
  "processing_options": {
    "parse_images": true,
    "extract_tables": true,
    "enable_ocr": true,
    "chunk_strategy": "semantic"
  }
}
```

**Response:**

```json
{
  "upload_id": "upload_789xyz012",
  "status": "processing",
  "documents": [
    {
      "document_id": "doc_456def789",
      "filename": "quarterly_report.pdf",
      "status": "processing",
      "size_bytes": 2048576,
      "pages": 24
    }
  ],
  "processing_time_estimate": "30s",
  "created_at": "2025-08-20T15:30:45Z"
}
```

### Get Document Status

```http
GET /api/v1/documents/{document_id}
```

**Response:**

```json
{
  "document_id": "doc_456def789",
  "filename": "quarterly_report.pdf",
  "status": "ready",
  "processing_details": {
    "chunks_created": 156,
    "tables_extracted": 8,
    "images_processed": 12,
    "processing_time": "28s"
  },
  "metadata": {
    "file_type": "pdf",
    "size_bytes": 2048576,
    "pages": 24,
    "language": "en",
    "created_at": "2025-08-20T15:30:45Z"
  }
}
```

### Delete Document

```http
DELETE /api/v1/documents/{document_id}
```

**Response:**

```json
{
  "document_id": "doc_456def789",
  "status": "deleted",
  "deleted_at": "2025-08-20T15:45:30Z"
}
```

### Search Documents

```http
POST /api/v1/documents/search
```

**Request Body:**

```json
{
  "query": "revenue growth quarterly comparison",
  "session_id": "sess_abc123def456",
  "filters": {
    "document_types": ["pdf", "docx"],
    "date_range": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-12-31T23:59:59Z"
    },
    "tags": ["financial", "quarterly"]
  },
  "options": {
    "similarity_top_k": 10,
    "enable_reranking": true,
    "min_relevance_score": 0.7,
    "include_content": true,
    "highlight_matches": true
  }
}
```

**Response:**

```json
{
  "query": "revenue growth quarterly comparison",
  "total_results": 8,
  "results": [
    {
      "document_id": "doc_456def789",
      "filename": "quarterly_report.pdf",
      "chunk_id": "chunk_789abc012",
      "relevance_score": 0.94,
      "content": "Revenue growth accelerated to 18% in Q4, compared to 12% in Q3...",
      "highlighted_content": "**Revenue growth** accelerated to 18% in Q4, compared to 12% in Q3...",
      "metadata": {
        "page": 3,
        "section": "Financial Performance",
        "word_count": 245
      }
    }
  ],
  "search_metadata": {
    "search_time": 0.28,
    "strategy_used": "hybrid_search",
    "reranking_applied": true,
    "filters_applied": 3
  }
}
```

## Analysis and Query Processing

### Analyze Documents

```http
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
    "context_optimization": true,
    "include_confidence_scores": true,
    "include_agent_trace": false
  },
  "filters": {
    "document_ids": ["doc_456def789", "doc_123abc456"],
    "date_range": {
      "start": "2024-10-01T00:00:00Z",
      "end": "2024-12-31T23:59:59Z"
    }
  }
}
```

**Response:**

```json
{
  "request_id": "req_789xyz012",
  "status": "completed",
  "analysis_result": {
    "content": "Based on the comprehensive analysis of Q4 2024 financial data, several key trends emerge:\n\n1. **Accelerated Revenue Growth**: Revenue increased 18% quarter-over-quarter, marking the strongest performance in 2024.\n\n2. **Margin Improvement**: Gross margins expanded to 42.3%, up from 39.8% in Q3, driven by operational efficiencies.\n\n3. **International Expansion**: International markets contributed 34% of total revenue, up from 28% in the previous quarter.\n\nKey performance indicators show consistent improvement across all business units, with particular strength in the technology and healthcare sectors.",
    "confidence_score": 0.89,
    "citations": [
      {
        "source": "quarterly_report.pdf",
        "page": 3,
        "relevance": 0.94,
        "excerpt": "Revenue growth accelerated to 18% in Q4..."
      },
      {
        "source": "quarterly_report.pdf",
        "page": 8,
        "relevance": 0.87,
        "excerpt": "Gross margins expanded to 42.3%..."
      }
    ],
    "key_insights": [
      "Revenue growth of 18% quarter-over-quarter",
      "Improved profit margins in Q4",
      "Strong performance in international markets"
    ]
  },
  "performance_metrics": {
    "total_time": 2.45,
    "agent_coordination_time": 0.32,
    "retrieval_time": 0.89,
    "synthesis_time": 1.24,
    "token_usage": {
      "input_tokens": 3456,
      "output_tokens": 578
    }
  },
  "timestamp": "2025-08-20T15:30:45Z"
}
```

### Stream Analysis

```http
POST /api/v1/analyze/stream
```

**Request Body:** (Same as `/analyze`)

**Response:** Server-Sent Events (SSE)

```
data: {"type": "agent_start", "agent": "query_router", "task": "Analyzing query complexity"}

data: {"type": "agent_complete", "agent": "query_router", "duration": 0.15}

data: {"type": "agent_start", "agent": "retrieval_expert", "task": "Searching documents"}

data: {"type": "response_chunk", "content": "Based on the comprehensive analysis"}

data: {"type": "response_chunk", "content": " of Q4 2024 financial data, several"}

data: {"type": "complete", "total_time": 2.45, "confidence": 0.89}
```

### Get Analysis Status

```http
GET /api/v1/analyze/{request_id}/status
```

**Response:**

```json
{
  "request_id": "req_789xyz012",
  "status": "completed",
  "progress": {
    "current_stage": "synthesis",
    "stages_completed": [
      {"stage": "routing", "duration": 0.15},
      {"stage": "retrieval", "duration": 0.89},
      {"stage": "synthesis", "duration": 1.24}
    ],
    "estimated_completion": null
  },
  "result_available": true,
  "created_at": "2025-08-20T15:30:45Z"
}
```

## Session Management

### Create Session

```http
POST /api/v1/sessions
```

**Request Body:**

```json
{
  "name": "Financial Analysis Session",
  "description": "Q4 2024 financial document analysis",
  "settings": {
    "context_window_size": 131072,
    "enable_gpu_acceleration": true,
    "performance_mode": "balanced"
  }
}
```

**Response:**

```json
{
  "session_id": "sess_abc123def456",
  "name": "Financial Analysis Session",
  "description": "Q4 2024 financial document analysis",
  "status": "active",
  "settings": {
    "context_window_size": 131072,
    "enable_gpu_acceleration": true,
    "performance_mode": "balanced"
  },
  "created_at": "2025-08-20T15:30:45Z"
}
```

### Get Session

```http
GET /api/v1/sessions/{session_id}
```

**Response:**

```json
{
  "session_id": "sess_abc123def456",
  "name": "Financial Analysis Session",
  "description": "Q4 2024 financial document analysis",
  "status": "active",
  "statistics": {
    "document_count": 3,
    "message_count": 12,
    "token_usage": 45672,
    "context_utilization": 0.35
  },
  "created_at": "2025-08-20T15:30:45Z",
  "last_activity": "2025-08-20T16:15:22Z"
}
```

### Delete Session

```http
DELETE /api/v1/sessions/{session_id}
```

**Response:**

```json
{
  "session_id": "sess_abc123def456",
  "status": "deleted",
  "deleted_at": "2025-08-20T16:30:15Z"
}
```

### Get Session History

```http
GET /api/v1/sessions/{session_id}/history
```

**Query Parameters:**
- `limit`: Number of messages to return (default: 50)
- `offset`: Offset for pagination (default: 0)
- `include_system`: Include system messages (default: false)

**Response:**

```json
{
  "session_id": "sess_abc123def456",
  "total_messages": 12,
  "messages": [
    {
      "message_id": "msg_456def789",
      "role": "user",
      "content": "Analyze the quarterly performance trends",
      "timestamp": "2025-08-20T15:30:45Z"
    },
    {
      "message_id": "msg_789abc012",
      "role": "assistant",
      "content": "Based on the comprehensive analysis of quarterly data...",
      "confidence": 0.89,
      "citations": [
        {
          "source": "quarterly_report.pdf",
          "page": 3,
          "relevance": 0.94
        }
      ],
      "timestamp": "2025-08-20T15:30:50Z"
    }
  ],
  "pagination": {
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

## Agent System

### Coordinate Agents

```http
POST /api/v1/agents/coordinate
```

**Request Body:**

```json
{
  "query": "Compare Q3 vs Q4 performance across all business units",
  "session_id": "sess_abc123def456",
  "coordination_options": {
    "enable_parallel_execution": true,
    "max_agent_timeout": 300,
    "enable_fallback": true,
    "include_agent_trace": true
  },
  "agent_preferences": {
    "query_router": {
      "temperature": 0.1,
      "confidence_threshold": 0.8
    },
    "retrieval_expert": {
      "top_k": 20,
      "enable_reranking": true
    },
    "synthesis_agent": {
      "max_length": 2048,
      "include_citations": true
    }
  }
}
```

**Response:**

```json
{
  "coordination_id": "coord_123abc456",
  "status": "completed",
  "result": {
    "content": "Comprehensive comparison analysis of Q3 vs Q4 performance...",
    "confidence": 0.91,
    "citations": [...],
    "key_insights": [...]
  },
  "agent_trace": [
    {
      "agent": "query_router",
      "action": "route_query",
      "input": "Compare Q3 vs Q4 performance across all business units",
      "output": {
        "strategy": "comparative_analysis",
        "complexity": "high",
        "estimated_time": 180
      },
      "duration": 0.12,
      "status": "completed"
    },
    {
      "agent": "planning_agent",
      "action": "create_execution_plan",
      "input": {...},
      "output": {
        "sub_tasks": [
          {"description": "Retrieve Q3 financial data", "priority": "high"},
          {"description": "Retrieve Q4 financial data", "priority": "high"},
          {"description": "Compare performance metrics", "priority": "medium"}
        ]
      },
      "duration": 0.08,
      "status": "completed"
    },
    {
      "agent": "retrieval_expert",
      "action": "retrieve_documents",
      "input": {...},
      "output": {
        "documents_found": 15,
        "relevance_scores": [0.94, 0.87, 0.82]
      },
      "duration": 1.23,
      "status": "completed"
    }
  ],
  "performance_metrics": {
    "total_coordination_time": 2.89,
    "parallel_execution_efficiency": 0.73,
    "agents_used": 5,
    "cache_hit_rate": 0.42
  }
}
```

### Get Agent Status

```http
GET /api/v1/agents/status
```

**Response:**

```json
{
  "system_status": "healthy",
  "agents": {
    "query_router": {
      "status": "active",
      "uptime": 3600,
      "last_activity": "2025-08-20T16:15:22Z",
      "performance": {
        "avg_response_time": 0.15,
        "success_rate": 0.98,
        "total_requests": 245
      }
    },
    "retrieval_expert": {
      "status": "active",
      "uptime": 3600,
      "last_activity": "2025-08-20T16:15:18Z",
      "performance": {
        "avg_response_time": 1.23,
        "success_rate": 0.95,
        "total_requests": 198
      }
    },
    "planning_agent": {
      "status": "active",
      "uptime": 3600,
      "last_activity": "2025-08-20T16:14:45Z",
      "performance": {
        "avg_response_time": 0.08,
        "success_rate": 0.99,
        "total_requests": 167
      }
    },
    "synthesis_agent": {
      "status": "active",
      "uptime": 3600,
      "last_activity": "2025-08-20T16:15:20Z",
      "performance": {
        "avg_response_time": 1.45,
        "success_rate": 0.97,
        "total_requests": 203
      }
    },
    "validation_agent": {
      "status": "active",
      "uptime": 3600,
      "last_activity": "2025-08-20T16:15:21Z",
      "performance": {
        "avg_response_time": 0.67,
        "success_rate": 0.96,
        "total_requests": 189
      }
    }
  },
  "resource_usage": {
    "vram_usage_gb": 13.8,
    "vram_total_gb": 16.0,
    "gpu_utilization": 78,
    "context_utilization": 0.35,
    "max_context": 131072
  }
}
```

### Execute Specific Agent

```http
POST /api/v1/agents/{agent_id}/execute
```

**Request Body:**

```json
{
  "action": "route_query",
  "input": {
    "query": "What are the key performance indicators for Q4?",
    "context": {
      "conversation_history": [...],
      "document_metadata": {...}
    }
  },
  "options": {
    "temperature": 0.1,
    "confidence_threshold": 0.8,
    "timeout": 30
  }
}
```

**Response:**

```json
{
  "execution_id": "exec_456def789",
  "agent": "query_router",
  "action": "route_query",
  "status": "completed",
  "result": {
    "strategy": "hybrid_search",
    "confidence": 0.89,
    "reasoning": "Query requires financial data retrieval and analysis",
    "parameters": {
      "search_strategy": "hybrid",
      "top_k": 15,
      "enable_reranking": true
    }
  },
  "performance": {
    "execution_time": 0.12,
    "memory_used": "45MB",
    "gpu_time": 0.08
  },
  "timestamp": "2025-08-20T16:15:22Z"
}
```

## System Monitoring

### Get System Metrics

```http
GET /api/v1/metrics/system
```

**Response:**

```json
{
  "timestamp": "2025-08-20T16:15:22Z",
  "system_health": "healthy",
  "gpu": {
    "available": true,
    "model": "RTX 4090",
    "utilization_percent": 78,
    "memory_used_gb": 13.8,
    "memory_total_gb": 16.0,
    "temperature_celsius": 72,
    "power_draw_watts": 285
  },
  "model_performance": {
    "model_name": "Qwen/Qwen3-4B-Instruct-2507-FP8",
    "avg_decode_tps": 145.2,
    "avg_prefill_tps": 1150.8,
    "context_utilization": 0.35,
    "cache_hit_rate": 0.42
  },
  "system_resources": {
    "cpu_usage_percent": 45,
    "ram_used_gb": 18.4,
    "ram_total_gb": 32.0,
    "disk_usage_percent": 67,
    "network_io": {
      "bytes_sent": 1048576,
      "bytes_received": 2097152
    }
  },
  "service_status": {
    "ollama": "running",
    "qdrant": "running",
    "vector_store": "ready",
    "embedding_model": "loaded"
  }
}
```

### Get Performance Metrics

```http
GET /api/v1/metrics/performance
```

**Query Parameters:**
- `start_time`: ISO timestamp for start of time range
- `end_time`: ISO timestamp for end of time range
- `interval`: Aggregation interval (1m, 5m, 1h)
- `metrics`: Comma-separated metric names

**Response:**

```json
{
  "time_range": {
    "start": "2025-08-20T15:15:22Z",
    "end": "2025-08-20T16:15:22Z",
    "interval": "5m"
  },
  "metrics": {
    "decode_tps": [
      {"timestamp": "2025-08-20T15:15:00Z", "value": 142.1},
      {"timestamp": "2025-08-20T15:20:00Z", "value": 145.8},
      {"timestamp": "2025-08-20T15:25:00Z", "value": 143.9}
    ],
    "vram_usage_gb": [
      {"timestamp": "2025-08-20T15:15:00Z", "value": 13.5},
      {"timestamp": "2025-08-20T15:20:00Z", "value": 13.7},
      {"timestamp": "2025-08-20T15:25:00Z", "value": 13.8}
    ],
    "agent_coordination_time": [
      {"timestamp": "2025-08-20T15:15:00Z", "value": 0.15},
      {"timestamp": "2025-08-20T15:20:00Z", "value": 0.12},
      {"timestamp": "2025-08-20T15:25:00Z", "value": 0.18}
    ]
  },
  "summary": {
    "decode_tps": {"avg": 143.9, "min": 142.1, "max": 145.8},
    "vram_usage_gb": {"avg": 13.67, "min": 13.5, "max": 13.8},
    "agent_coordination_time": {"avg": 0.15, "min": 0.12, "max": 0.18}
  }
}
```

### Health Check

```http
GET /api/v1/health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-08-20T16:15:22Z",
  "uptime": 3600,
  "checks": {
    "database": {
      "status": "healthy",
      "response_time": 0.003
    },
    "gpu": {
      "status": "healthy",
      "available": true,
      "memory_usage": 0.8625
    },
    "models": {
      "status": "healthy",
      "llm_loaded": true,
      "embedding_loaded": true,
      "reranker_loaded": true
    },
    "agents": {
      "status": "healthy",
      "active_agents": 5,
      "avg_response_time": 0.65
    }
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
| `VALIDATION_ERROR` | Request validation failed | 400 |
| `PROCESSING_ERROR` | Document processing failed | 422 |
| `AUTHENTICATION_ERROR` | Invalid or missing authentication | 401 |
| `AUTHORIZATION_ERROR` | Insufficient permissions | 403 |

### Error Details

#### Validation Error Example

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field_errors": [
        {
          "field": "query",
          "error": "Query cannot be empty",
          "value": ""
        },
        {
          "field": "options.max_response_tokens",
          "error": "Must be between 1 and 4096",
          "value": 8192
        }
      ]
    },
    "request_id": "req_789xyz012",
    "timestamp": "2025-08-20T15:30:45Z"
  }
}
```

#### Processing Error Example

```json
{
  "error": {
    "code": "PROCESSING_ERROR",
    "message": "Document processing failed",
    "details": {
      "document_id": "doc_456def789",
      "filename": "corrupted_file.pdf",
      "processing_stage": "text_extraction",
      "underlying_error": "PDF parsing error: Invalid PDF structure"
    },
    "request_id": "req_789xyz012",
    "timestamp": "2025-08-20T15:30:45Z"
  }
}
```

## Rate Limiting

### Request Limits

| Endpoint | Limit | Window |
|----------|--------|--------|
| `/analyze` | 60 requests | 1 hour |
| `/analyze/stream` | 30 requests | 1 hour |
| `/documents/upload` | 10 requests | 1 hour |
| `/documents/search` | 300 requests | 1 hour |
| `/agents/coordinate` | 120 requests | 1 hour |
| `/metrics/*` | 1000 requests | 1 hour |

### Rate Limit Headers

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1692537845
X-RateLimit-Window: 3600
```

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded for endpoint /api/v1/analyze",
    "details": {
      "limit": 60,
      "window": 3600,
      "reset_time": "2025-08-20T17:00:00Z"
    },
    "request_id": "req_789xyz012",
    "timestamp": "2025-08-20T15:30:45Z"
  }
}
```

## SDK Examples

### Python SDK

```python
from docmind_sdk import DocMindClient

# Initialize client
client = DocMindClient(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    session_id="sess_abc123def456"
)

# Simple analysis
result = await client.analyze(
    query="Summarize the key financial trends",
    options={"enable_citations": True}
)

print(f"Response: {result.analysis_result.content}")
print(f"Confidence: {result.analysis_result.confidence_score:.2f}")

# Streaming analysis
async for chunk in client.analyze_stream(
    query="Detailed market analysis with recommendations"
):
    if chunk.type == "response_chunk":
        print(chunk.content, end="")
    elif chunk.type == "complete":
        print(f"\nCompleted in {chunk.total_time}s")

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

for result in search_results.results:
    print(f"📄 {result.filename} (Score: {result.relevance_score:.3f})")
    print(f"   {result.content[:100]}...")
```

### JavaScript/TypeScript SDK

```javascript
import { DocMindClient } from '@docmind/sdk';

const client = new DocMindClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key',
  sessionId: 'sess_abc123def456'
});

// Analysis with error handling
try {
  const result = await client.analyze({
    query: 'Analyze competitive landscape',
    options: { performanceMode: 'balanced' }
  });
  
  console.log(result.analysisResult.content);
  console.log(`Confidence: ${result.analysisResult.confidenceScore}`);
  
} catch (error) {
  if (error.code === 'CONTEXT_OVERFLOW') {
    console.log('Query too long, trying with shorter context');
  } else if (error.code === 'RATE_LIMIT_EXCEEDED') {
    console.log(`Rate limit exceeded. Reset at: ${error.details.resetTime}`);
  } else {
    console.error('Analysis failed:', error.message);
  }
}

// Real-time metrics monitoring
const metrics = await client.getSystemMetrics();
console.log(`GPU Usage: ${metrics.gpu.utilizationPercent}%`);
console.log(`VRAM: ${metrics.gpu.memoryUsedGb}GB / ${metrics.gpu.memoryTotalGb}GB`);

// Session management
const session = await client.createSession({
  name: 'Market Analysis',
  settings: {
    performanceMode: 'thorough',
    enableGpuAcceleration: true
  }
});

console.log(`Created session: ${session.sessionId}`);
```

### cURL Examples

#### Upload Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "files": [{
      "name": "report.pdf",
      "content": "base64_encoded_content",
      "content_type": "application/pdf"
    }],
    "session_id": "sess_abc123def456",
    "processing_options": {
      "parse_images": true,
      "extract_tables": true
    }
  }'
```

#### Analyze Documents

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key financial trends?",
    "session_id": "sess_abc123def456",
    "options": {
      "enable_citations": true,
      "performance_mode": "balanced"
    }
  }'
```

#### Stream Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/stream" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "query": "Comprehensive quarterly analysis",
    "session_id": "sess_abc123def456"
  }'
```

## Best Practices

### Optimal Usage Patterns

1. **Session Management**: Reuse sessions for related queries to benefit from context preservation
2. **Performance Mode Selection**: 
   - `fast`: Simple queries, quick turnaround
   - `balanced`: Most use cases, good speed/quality balance
   - `thorough`: Complex analysis requiring deep reasoning
3. **Context Optimization**: Enable for long conversations to maintain relevance
4. **Caching**: Structure similar queries to benefit from system caching
5. **Error Handling**: Always implement retry logic with exponential backoff

### Performance Optimization

1. **Batch Operations**: Group multiple document uploads and searches
2. **Streaming**: Use streaming endpoints for long responses to improve perceived performance
3. **Filter Usage**: Apply document filters to reduce retrieval scope and improve speed
4. **Context Management**: Regularly clear conversation context for long sessions
5. **Resource Monitoring**: Monitor system metrics to optimize hardware usage

### Security Considerations

1. **API Key Management**: Rotate API keys regularly and store securely
2. **Session Isolation**: Use separate sessions for different users/contexts
3. **Input Validation**: Validate all inputs before sending to API
4. **Rate Limiting**: Implement client-side rate limiting to avoid 429 errors
5. **Error Information**: Don't expose sensitive system information in error messages

### Integration Patterns

#### Webhook Integration

```javascript
// Set up webhook for long-running analysis
const result = await client.analyze({
  query: "Complex multi-document analysis",
  options: {
    webhook_url: "https://your-app.com/api/analysis-complete",
    webhook_headers: {
      "Authorization": "Bearer webhook-secret"
    }
  }
});

console.log(`Analysis started: ${result.requestId}`);

// Webhook handler (Express.js example)
app.post('/api/analysis-complete', (req, res) => {
  const { requestId, status, result } = req.body;
  
  if (status === 'completed') {
    console.log(`Analysis ${requestId} completed`);
    console.log(result.analysisResult.content);
  }
  
  res.status(200).send('OK');
});
```

#### Batch Processing

```python
# Process multiple queries in batch
queries = [
    "Summarize Q1 performance",
    "Analyze Q2 trends", 
    "Compare Q3 vs Q4"
]

# Submit all queries
tasks = []
for query in queries:
    result = await client.analyze(
        query=query,
        options={"async": True}  # Run asynchronously
    )
    tasks.append(result.request_id)

# Poll for completion
for task_id in tasks:
    while True:
        status = await client.get_analysis_status(task_id)
        if status.status == 'completed':
            result = await client.get_analysis_result(task_id)
            print(f"Query completed: {result.analysis_result.content[:100]}...")
            break
        elif status.status == 'failed':
            print(f"Query failed: {status.error}")
            break
        else:
            await asyncio.sleep(1)  # Wait before polling again
```

## Migration from Python API

For existing Python API users, migration to REST API is straightforward:

### Before (Python API)

```python
from src.agents.coordinator import MultiAgentCoordinator
from src.utils.document import load_documents_unstructured

coordinator = MultiAgentCoordinator()
documents = await load_documents_unstructured(file_paths, settings)
response = coordinator.process_query(query, context=documents)
```

### After (REST API)

```python
from docmind_sdk import DocMindClient

client = DocMindClient(base_url="http://localhost:8000")

# Upload documents
upload_result = await client.upload_documents(file_paths)
session_id = upload_result.session_id

# Analyze query
response = await client.analyze(
    query=query,
    session_id=session_id
)
```

## Summary

The DocMind AI REST API provides comprehensive HTTP endpoints for:

- **Document Management**: Upload, process, search, and manage documents
- **Analysis & Queries**: Multi-agent coordination for intelligent document analysis
- **Session Management**: Persistent conversation context and memory
- **Agent System**: Direct access to individual agents and coordination
- **System Monitoring**: Performance metrics and health checks
- **Error Handling**: Comprehensive error codes and troubleshooting

**Key Features:**

- Full parity with Python API functionality
- Streaming support for real-time responses
- Comprehensive error handling and validation
- Performance monitoring and metrics
- Rate limiting and security features
- Multi-language SDK support

**Development Status:**

The REST API is currently under active development with a planned release in Q2 2025. Beta access will be available for early adopters and enterprise customers.

For current functionality, use the [Internal Python API](./internal-api.md). For updates on REST API development progress, visit the [project repository](https://github.com/BjornMelin/docmind-ai-llm).