"""Pydantic models for DocMind AI - Compatibility Layer.

This module provides backward compatibility for the refactored settings system.
The monolithic AppSettings has been split into modular
groups for better maintainability.

**MIGRATION NOTICE**:
The settings system has been refactored into modular groups. For new code, use:

    from models import AnalysisOutput
    from models.settings import settings

    # New modular API
    model = settings.embedding.dense_embedding_model
    gpu_enabled = settings.gpu.gpu_acceleration

For backward compatibility, the old API still works:

    from models import AppSettings  # Compatibility wrapper
    settings = AppSettings()
    model = settings.dense_embedding_model  # Still works!

Classes:
    AnalysisOutput: Structured schema for document analysis results.
    AppSettings: Backward compatibility wrapper
(use models.settings.settings for new code).
"""

# Backward compatibility imports
from models.analysis import AnalysisOutput
from models.settings.migration import AppSettingsCompat

# For backward compatibility, expose AppSettings as the old monolithic class
AppSettings = AppSettingsCompat


# Re-export AnalysisOutput for backward compatibility
# (The actual implementation is now in models/analysis.py)
__all__ = ["AnalysisOutput", "AppSettings"]


# The old monolithic AppSettings class has been refactored into modular components.
# This file now provides backward compatibility through the AppSettingsCompat wrapper.
# For new code, use: from models.settings import settings

# Example of the new modular API:
# from models.settings import settings
# model = settings.embedding.dense_embedding_model
# gpu_enabled = settings.gpu.gpu_acceleration
