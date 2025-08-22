"""System tests for DocMind AI full pipeline validation.

This package contains comprehensive system tests that validate the complete
DocMind AI pipeline with real models and GPU acceleration. These tests are
designed to run on GPU-enabled systems and test actual functionality with
minimal mocking.

Test Categories:
    test_full_pipeline.py: Full GPU pipeline validation with real models
    test_model_loading.py: Model loading and resource management tests

System tests are marked with @pytest.mark.system and require GPU resources.
They are automatically skipped on systems without GPU or required models.
"""
