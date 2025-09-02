# DocMind AI API

## Overview

DocMind exposes both a REST API and a Python API. This document consolidates core information for both. Examples are provided in `docs/api/examples/`.

## REST API (Public)

- Base URL: `http://localhost:8000/api/v1`
- Authentication: `Authorization: Bearer <api-key>` or `X-Session-ID: <session>`

### Documents

- Upload: `POST /api/v1/documents/upload`
- Status: `GET /api/v1/documents/{document_id}`
- Delete: `DELETE /api/v1/documents/{document_id}`
- Search: `POST /api/v1/documents/search`

Example (upload request body): see `examples/upload-request.json`

### Analysis

- Analyze: `POST /api/v1/analyze`
- Stream: `POST /api/v1/analyze/stream` (SSE)
- Status: `GET /api/v1/analyze/{request_id}/status`

Example (analyze request): see `examples/analyze-request.json`

### Sessions

- Create: `POST /api/v1/sessions`
- Get: `GET /api/v1/sessions/{session_id}`
- History: `GET /api/v1/sessions/{session_id}/history`

### Agent Coordination

- Coordinate: `POST /api/v1/agents/coordinate`

Notes

- Payloads may evolve; keep clients tolerant of additional fields.
- Use SSE for streaming responses when interactive output is needed.

## Python API (Internal)

### Quick Start

```python
from src.agents.coordinator import MultiAgentCoordinator
from src.utils.document import load_documents_unstructured
from src.config import settings

coordinator = MultiAgentCoordinator()
documents = await load_documents_unstructured(["/path/to/document.pdf"], settings)
response = coordinator.process_query(
    "Analyze the key insights from this document",
    context=documents,
)
print(response.content)
```

### Configuration

```python
from src.config import settings
print(settings.vllm.model)
settings.agents.decision_timeout = 300
settings.processing.chunk_size = 2000
```

### Environment

```bash
# Core
DOCMIND_LLM__MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
DOCMIND_VLLM__CONTEXT_WINDOW=131072
DOCMIND_LLM__BASE_URL=http://localhost:11434

# Agents
DOCMIND_AGENTS__ENABLE_MULTI_AGENT=true
DOCMIND_AGENTS__DECISION_TIMEOUT=200

# Performance
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_KV_CACHE_DTYPE=fp8_e5m2
VLLM_GPU_MEMORY_UTILIZATION=0.85
```

## Examples

- Upload: `docs/api/examples/upload-request.json`
- Upload response: `docs/api/examples/upload-response.json`
- Analyze: `docs/api/examples/analyze-request.json`
- Analyze response: `docs/api/examples/analyze-response.json`
- Python usage: `docs/api/examples/python-example.py`
