# Requirements Traceability Matrix (RTM)

| ID | Title | Source | ADR(s) | Code file(s) | Test(s) | Verification | Status |
|----|-------|--------|--------|--------------|---------|--------------|--------|
| FR-001 | Unstructured ingest auto+OCR | ADR-002 | 002 | src/processing/document_processor.py | tests_ingest/* | demonstration | Planned |
| FR-002 | LlamaIndex pipeline cache | ADR-010 | 010 | src/processing/document_processor.py | tests_ingest/* | test+analysis | Planned |
| FR-004 | BGE-M3 + OpenCLIP/SigLIP | ADR-004 | 004 | src/retrieval/embeddings.py | tests_embed/* | test | Planned |
| FR-005 | Qdrant named vectors hybrid | ADR-005/006 | 005,006 | src/retrieval/query_engine.py | tests_retrieval/* | test | Planned |
| FR-007 | BGE rerank + ColPali | ADR-007 | 007 | src/retrieval/reranking.py | tests_rerank/* | test | Planned |
| FR-009 | GraphRAG PropertyGraphIndex | ADR-008 | 008 | src/retrieval/graph_config.py | tests_graph/* | test | Planned |
| FR-010 | Streamlit multipage | ADR-012 | 012 | src/app.py; src/pages/* | ui_tests/* | inspection | In repo |
| FR-012 | Multi-provider UI | ADR-009 | 009 | src/config/llm_factory.py; src/pages/settings.py | ui_tests/* | demo | In repo |
| FR-014 | LangGraph supervisor | ADR-001 | 001 | src/agents/* | tests_agents/* | test | In repo |
