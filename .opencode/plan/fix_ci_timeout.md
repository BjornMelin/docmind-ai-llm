# Plan: Fix CI Timeout in Settings Page Test

## Issue
The test `test_settings_page_renders_and_has_hybrid_toggle` in `src/pages/04_settings.py` is failing due to a timeout.
Importing the settings page triggers a chain of blocking imports (LlamaIndex, PyTorch) that exceed the test timeout on CI.

## Proposed Changes

### 1. `src/retrieval/adapter_registry.py`
- **Action:** Comment out or remove the eager module-level call to `ensure_default_adapter()`.
- **Reason:** This forces LlamaIndex initialization on import. Accessors like `get_adapter()` already call it lazily.

### 2. `src/config/integrations.py`
- **Action:** Move top-level imports of `src.models.embeddings` inside `initialize_integrations` and `get_unified_embedder`.
- **Reason:** `src.models.embeddings` imports PyTorch, which is very slow to load.

### 3. `src/ui/components/provider_badge.py`
- **Action:** Move the import of `src.retrieval.adapter_registry` inside the `provider_badge` function.
- **Reason:** Prevents eager loading of the registry (and thus LlamaIndex) when this component is imported.

## Verification
- Run `uv run pytest tests/integration/test_settings_app_test.py` to confirm the test passes and potentially runs faster.
- Run `uv run pytest` to ensure no regressions in other tests.
