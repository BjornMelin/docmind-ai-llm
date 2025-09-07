# Requirements Traceability Matrix (RTM)

| ID | Title | Source | ADR(s) | Code file(s) | Test(s) | Verification | Status |
|----|-------|--------|--------|--------------|---------|--------------|--------|
| FR-001 | Unstructured ingest auto+OCR | ADR-002 | 002 | src/processing/document_processor.py; src/processing/utils.py | tests/unit/processing/test_document_processor_unit.py; tests/unit/processing/test_unstructured_transformation_config.py; tests/integration/test_ingestion_pipeline_pdf_images.py | test | Completed |
| FR-002 | LlamaIndex pipeline cache | ADR-010 | 010 | src/processing/document_processor.py | tests/unit/cache/test_ingestion_cache.py; tests/integration/test_ingestion_pipeline_pdf_images.py | test+analysis | Completed |
| FR-003 | Deterministic IDs + pdf_page_image | ADR-002 | 002 | src/processing/document_processor.py; src/processing/pdf_pages.py; src/models/schemas.py | tests/unit/processing/test_pdf_pages_unit.py; tests/unit/processing/test_deterministic_ids_unit.py; tests/integration/test_ingestion_pipeline_pdf_images.py | test | Completed |
| FR-004 | BGE‑M3 + SigLIP default (OpenCLIP optional) | ADR‑002 | 002 | src/retrieval/bge_m3_index.py; src/config/integrations.py; src/models/embeddings.py; src/utils/multimodal.py | tests/unit/retrieval/test_embeddings_refactored.py; tests/unit/models/test_bge_m3_text_embedder_unit.py; tests/unit/models/test_image_embedder_unit.py; tests/integration/test_unified_embeddings_in_retrieval_integration.py | test | Completed |
| FR-005 | Qdrant server‑side hybrid (RRF default) | ADR‑005/006 | 005,006 | src/retrieval/query_engine.py | tests/integration/test_hybrid_retrieval_qdrant.py | test | Planned |
| FR-007 | BGE text rerank + SigLIP visual default (optional ColPali) | ADR‑037 | 037 | src/retrieval/reranking.py | tests/integration/test_query_engine_mm.py; tests_rerank/* | test | Planned |
| FR-008 | Always-on hybrid + reranking (no UI toggles; env overrides only) | ADR‑024/036 | 024,036 | src/retrieval/query_engine.py; src/retrieval/reranking.py; src/pages/04_settings.py | tests/integration/test_settings_page.py | inspection | Planned |
| FR-017 | Minimal telemetry (latency, fusion mode, reranker hits) | ADR‑032 | 032 | src/retrieval/query_engine.py; src/retrieval/reranking.py | tests/unit/test_telemetry_assertions.py | test | Planned |
| FR-009 | GraphRAG PropertyGraphIndex | ADR-008 | 008 | src/retrieval/graph_config.py | tests_graph/* | test | Planned |
| FR-010 | Streamlit multipage | ADR-012 | 012 | src/app.py; src/pages/04_settings.py | tests/unit/test_integrations_runtime.py; tests/integration/test_settings_page.py | inspection | Completed |
| FR-012 | Multi-provider UI | ADR-009 | 009 | src/config/llm_factory.py; src/pages/04_settings.py | tests/unit/test_llm_factory.py; tests/unit/test_llm_factory_extended.py; tests/unit/test_integrations_runtime.py; tests/integration/test_settings_page.py | test | Completed |
| FR-014 | LangGraph supervisor | ADR-001 | 001 | src/agents/* | tests_agents/* | test | In repo |
