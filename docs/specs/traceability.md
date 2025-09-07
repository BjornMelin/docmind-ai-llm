# Requirements Traceability Matrix (RTM)

| ID | Title | Source | ADR(s) | Code file(s) | Test(s) | Verification | Status |
|----|-------|--------|--------|--------------|---------|--------------|--------|
| FR-001 | Unstructured ingest auto+OCR | ADR-002 | 002 | src/processing/document_processor.py; src/processing/utils.py | tests/unit/processing/test_document_processor_unit.py; tests/unit/processing/test_unstructured_transformation_config.py; tests/integration/test_ingestion_pipeline_pdf_images.py | test | Completed |
| FR-002 | LlamaIndex pipeline cache | ADR-010 | 010 | src/processing/document_processor.py | tests/unit/cache/test_ingestion_cache.py; tests/integration/test_ingestion_pipeline_pdf_images.py | test+analysis | Completed |
| FR-003 | Deterministic IDs + pdf_page_image | ADR-002 | 002 | src/processing/document_processor.py; src/processing/pdf_pages.py; src/models/schemas.py | tests/unit/processing/test_pdf_pages_unit.py; tests/unit/processing/test_deterministic_ids_unit.py; tests/integration/test_ingestion_pipeline_pdf_images.py | test | Completed |
| FR-004 | BGE-M3 + OpenCLIP/SigLIP | ADR-004 | 004 | src/models/embeddings.py; src/utils/multimodal.py; src/retrieval/embeddings.py | tests/unit/models/*; tests/unit/utils/test_multimodal.py; tests/integration/test_unified_embeddings_in_retrieval_integration.py | test | Completed |
| FR-005 | Qdrant named vectors hybrid | ADR-005/006 | 005,006 | src/retrieval/query_engine.py | tests_retrieval/* | test | Planned |
| FR-007 | BGE rerank + ColPali | ADR-007 | 007 | src/retrieval/reranking.py | tests_rerank/* | test | Planned |
| FR-009 | GraphRAG PropertyGraphIndex | ADR-008 | 008 | src/retrieval/graph_config.py | tests_graph/* | test | Planned |
| FR-010 | Streamlit multipage | ADR-012 | 012 | src/app.py; src/pages/04_settings.py | tests/unit/test_integrations_runtime.py; tests/integration/test_settings_page.py | inspection | Completed |
| FR-012 | Multi-provider UI | ADR-009 | 009 | src/config/llm_factory.py; src/pages/04_settings.py | tests/unit/test_llm_factory.py; tests/unit/test_llm_factory_extended.py; tests/unit/test_integrations_runtime.py; tests/integration/test_settings_page.py | test | Completed |
| FR-014 | LangGraph supervisor | ADR-001 | 001 | src/agents/* | tests_agents/* | test | In repo |
