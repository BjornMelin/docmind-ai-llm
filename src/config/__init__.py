"""DocMind AI Configuration Module.

This module provides centralized configuration management for DocMind AI,
including application settings, LlamaIndex setup, vLLM configuration,
and KV cache optimization.

Components:
    settings: Main application settings and configuration
    app_settings: Application-specific settings
    llamaindex_setup: LlamaIndex configuration and setup
    vllm_config: vLLM configuration for inference optimization
    kv_cache: KV cache configuration for memory optimization
"""

from src.config.settings import app_settings

__all__ = [
    "app_settings",
]
