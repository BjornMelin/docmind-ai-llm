# Testing Notes and Patch Seams

This document summarizes how tests should interact with production code, the
preferred patch seams, and stability tips for integration/UI tests.

## Patch the Real Consumer Seams

Avoid patching symbols from `src.app`. Instead, patch the concrete modules
consumed by the app and pages:

- LlamaIndex LLMs
  - `llama_index.llms.ollama.Ollama`
  - `llama_index.llms.llama_cpp.LlamaCPP`
  - `llama_index.llms.openai_like.OpenAILike`
- Vector stores / indexes
  - `llama_index.core.VectorStoreIndex`
  - `llama_index.core.vector_stores.SimpleVectorStore`
- Document loading
  - `src.utils.document.load_documents_unstructured`

Telemetry tests that target tools implemented as LangChain `StructuredTool`
should import the module using `importlib.import_module("src.agents.tools.router_tool")`
and patch `log_jsonl` on that module object, not on the `StructuredTool`
proxy.

## Streamlit AppTest stability

- Where pages depend on persisted state (e.g., the latest snapshot), tests
  should stub producers/consumers at the module level before constructing
  `AppTest`. This ensures import-time resolution picks up deterministic stubs.
- The Chat page tolerates missing snapshots and hydrates a minimal router to
  keep the page functional under tests.

## DuckDB KV store stability in tests

Some tests indirectly touch the DuckDB-backed KV store used by processing
pipelines. If you encounter intermittent crashes or environment-specific
segfaults from the DuckDB binary during testing:

- Prefer isolating KV file paths per test session (temporary directory); and/or
- Mock client creation on tests that do not require exercising DuckDB itself.

These mitigations reduce cross-test interference and OS/driver noise.
