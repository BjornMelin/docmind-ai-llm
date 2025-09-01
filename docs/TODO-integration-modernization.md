Integration Modernization TODO (2025-09-01)

Status
- Unit tier: green with coverage and lint artifacts.
- Integration tier: router + reranking tests partially stabilized; remaining work scheduled below.

Planned tasks

1) Router/Selector integration (tests/integration/test_query_engine.py)
- Ensure all router instantiations pass a deterministic LLM (llm=mock_llm_for_routing).
- Replace method-mocking of LLM.complete with MockLlamaIndexLLM.response_text assignment.

2) Reranking integration (tests/integration/test_query_engine_reranking_integration.py)
- Use normalize_scores=False in tests that assert exact float equality; otherwise assert ordering/monotonicity and lengths.
- Add mock_memory_monitor fixture (shared_fixtures) for memory efficiency tests.

3) Fixtures & isolation
- Add an integration-only autouse session fixture to force Settings.llm = MockLLM for integration runs (tests/integration/conftest.py) and restore after session.

4) Optional (deferred) src improvement
- src/retrieval/reranking.py: Avoid double-normalization if CrossEncoder already applies sigmoid via activation_fn when num_labels=1. Strategy:
  - Detect CrossEncoder activation / logits semantics and skip post-sigmoid when outputs are already probabilities.
  - Gate behind normalize_scores flag to preserve current behavior. Add unit coverage.

5) CI polish
- Run tests/integration in CI with offline settings; ensure no external network calls or GPU-required paths.
- Keep coverage reported for src/ only; integration tests can remain unmeasured if they cause flaky thresholds.

Acceptance
- tests/integration suite passes locally and in CI without network calls.
- No test-only shims in src/; all accommodations live under tests/.

