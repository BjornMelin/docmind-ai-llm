"""spaCy model management utilities for DocMind AI.

This module provides comprehensive spaCy model management including automatic
downloading, caching, and error handling. Consolidates spaCy-related functionality
to follow DRY principles and provide consistent NLP model management across
the application.

Key features:
- Automatic model downloading and installation
- Model validation and compatibility checking
- Caching and resource management
- Comprehensive error handling with fallbacks
- Support for multiple spaCy model sizes and languages

Example:
    Basic spaCy model management::

        from utils.spacy_utils import ensure_spacy_model, get_spacy_model_manager

        # Load model with automatic download if needed
        nlp = ensure_spacy_model("en_core_web_sm")
        if nlp:
            doc = nlp("This is a test sentence.")
            entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Use model manager for advanced features
        manager = get_spacy_model_manager()
        models_info = manager.get_available_models()
"""

import subprocess
import time
from typing import Any

from loguru import logger

from .exceptions import ConfigurationError
from .logging_utils import log_error_with_context, log_performance
from .retry_utils import with_fallback


@with_fallback(lambda model_name: None)  # Graceful fallback returns None
def ensure_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """Ensure spaCy model is available, download if needed.

    Attempts to load a spaCy model and automatically downloads it if not
    found locally. Provides robust error handling, structured logging,
    and graceful fallbacks throughout the process.

    Args:
        model_name: Name of the spaCy model to load. Common options include:
            - 'en_core_web_sm': Small English model (~15MB)
            - 'en_core_web_md': Medium English model (~40MB)
            - 'en_core_web_lg': Large English model (~560MB)
            - 'en_core_web_trf': Transformer English model (~560MB)
            Defaults to 'en_core_web_sm'.

    Returns:
        Loaded spaCy Language model instance ready for NLP processing,
        or None if loading fails and fallback is used.

    Raises:
        ConfigurationError: If the model cannot be loaded or downloaded, or if
            spaCy is not installed.

    Note:
        Downloads can take several minutes depending on model size and
        network speed. The function handles subprocess execution for
        model downloads and provides detailed logging throughout.

    Example:
        >>> nlp = ensure_spacy_model("en_core_web_sm")
        >>> if nlp:
        ...     doc = nlp("This is a test sentence.")
        ...     entities = [(ent.text, ent.label_) for ent in doc.ents]
        ...     print(f"Found entities: {entities}")
    """
    start_time = time.perf_counter()

    logger.info(f"Loading spaCy model: {model_name}")

    try:
        import spacy

        # Try to load existing model first
        try:
            nlp = spacy.load(model_name)
            logger.success(f"spaCy model '{model_name}' loaded successfully")

            duration = time.perf_counter() - start_time
            log_performance(
                "spacy_model_load",
                duration,
                model_name=model_name,
                loaded_from_cache=True,
            )
            return nlp

        except OSError:
            # Model not found locally, try to download
            logger.info(f"spaCy model '{model_name}' not found locally, downloading...")

            try:
                # Download model with timeout and error handling
                result = subprocess.run(
                    ["python", "-m", "spacy", "download", model_name],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout for downloads
                )

                logger.info(
                    f"spaCy model '{model_name}' downloaded successfully",
                    extra={
                        "download_output": result.stdout.strip()[:200]
                    },  # Log first 200 chars
                )

                # Try loading again after download
                nlp = spacy.load(model_name)
                logger.success(f"spaCy model '{model_name}' loaded after download")

                duration = time.perf_counter() - start_time
                log_performance(
                    "spacy_model_download_and_load",
                    duration,
                    model_name=model_name,
                    download_required=True,
                )
                return nlp

            except subprocess.TimeoutExpired as e:
                raise ConfigurationError(
                    f"spaCy model '{model_name}' download timed out",
                    context={
                        "model_name": model_name,
                        "timeout_seconds": 300,
                        "suggestion": "Try downloading manually or use a smaller model",
                    },
                    operation="spacy_model_download",
                ) from e
            except (subprocess.CalledProcessError, OSError) as e:
                error_context = {
                    "model_name": model_name,
                    "command": ["python", "-m", "spacy", "download", model_name],
                    "stderr": getattr(e, "stderr", str(e)),
                }

                log_error_with_context(e, "spacy_model_download", context=error_context)

                raise ConfigurationError(
                    f"Failed to download spaCy model '{model_name}'",
                    context=error_context,
                    original_error=e,
                    operation="spacy_model_download",
                ) from e

    except ImportError as e:
        log_error_with_context(
            e,
            "spacy_import",
            context={
                "model_name": model_name,
                "suggestion": "Install spaCy with: pip install spacy",
            },
        )

        raise ConfigurationError(
            "spaCy is not installed",
            context={
                "model_name": model_name,
                "installation_command": "pip install spacy",
            },
            original_error=e,
            operation="spacy_import",
        ) from e

    except Exception as e:
        log_error_with_context(
            e, "spacy_model_ensure", context={"model_name": model_name}
        )

        raise ConfigurationError(
            f"Unexpected error loading spaCy model '{model_name}'",
            context={"model_name": model_name},
            original_error=e,
            operation="spacy_model_ensure",
        ) from e


class SpacyModelManager:
    """Centralized manager for spaCy models with caching and validation.

    Provides comprehensive spaCy model management including model validation,
    caching, resource management, and automatic cleanup. Supports multiple
    models and languages with optimized loading and memory usage.
    """

    def __init__(self):
        """Initialize the spaCy model manager."""
        self._models = {}
        self._model_info = {}

    def get_model(
        self, model_name: str = "en_core_web_sm", force_reload: bool = False
    ) -> Any:
        """Get or load a spaCy model with caching.

        Args:
            model_name: Name of the spaCy model to load
            force_reload: Force model reload even if cached

        Returns:
            spaCy Language model instance or None if loading fails
        """
        if force_reload or model_name not in self._models:
            logger.info(f"Loading spaCy model: {model_name}")
            model = ensure_spacy_model(model_name)

            if model is not None:
                self._models[model_name] = model
                self._model_info[model_name] = {
                    "loaded_at": time.time(),
                    "language": model.lang if hasattr(model, "lang") else "unknown",
                    "pipeline": list(model.pipe_names)
                    if hasattr(model, "pipe_names")
                    else [],
                    "vocab_size": len(model.vocab) if hasattr(model, "vocab") else 0,
                }
                logger.info(f"Cached spaCy model: {model_name}")
            else:
                logger.warning(f"Failed to load spaCy model: {model_name}")

        return self._models.get(model_name)

    def validate_model(self, model_name: str) -> dict[str, Any]:
        """Validate a spaCy model and return detailed information.

        Args:
            model_name: Name of the model to validate

        Returns:
            Dictionary with validation results and model specifications
        """
        validation_result = {
            "model_name": model_name,
            "available": False,
            "language": None,
            "pipeline_components": [],
            "vocab_size": 0,
            "model_size": "unknown",
            "capabilities": {},
            "error": None,
        }

        try:
            # Attempt to load model for validation
            nlp = ensure_spacy_model(model_name)

            if nlp is not None:
                validation_result.update(
                    {
                        "available": True,
                        "language": nlp.lang,
                        "pipeline_components": list(nlp.pipe_names),
                        "vocab_size": len(nlp.vocab),
                    }
                )

                # Determine model size based on name
                if "sm" in model_name:
                    validation_result["model_size"] = "small"
                elif "md" in model_name:
                    validation_result["model_size"] = "medium"
                elif "lg" in model_name:
                    validation_result["model_size"] = "large"
                elif "trf" in model_name:
                    validation_result["model_size"] = "transformer"

                # Check capabilities
                validation_result["capabilities"] = {
                    "tokenization": True,  # Always available
                    "pos_tagging": "tagger" in nlp.pipe_names,
                    "lemmatization": "lemmatizer" in nlp.pipe_names,
                    "dependency_parsing": "parser" in nlp.pipe_names,
                    "named_entity_recognition": "ner" in nlp.pipe_names,
                    "sentence_segmentation": "sentencizer" in nlp.pipe_names
                    or "parser" in nlp.pipe_names,
                }

                logger.info(f"Model validation successful: {model_name}")

        except Exception as e:
            validation_result["error"] = str(e)
            logger.warning(f"Model validation failed for {model_name}: {e}")

        return validation_result

    def get_available_models(self) -> list[dict[str, Any]]:
        """Get list of commonly available spaCy models.

        Returns:
            List of dictionaries with model information and recommendations
        """
        common_models = [
            {
                "name": "en_core_web_sm",
                "language": "English",
                "size": "small",
                "download_size_mb": 15,
                "description": "Compact English model for basic NLP tasks",
                "recommended_for": ["development", "basic_nlp", "resource_constrained"],
            },
            {
                "name": "en_core_web_md",
                "language": "English",
                "size": "medium",
                "download_size_mb": 40,
                "description": "Medium English model with word vectors",
                "recommended_for": [
                    "production",
                    "similarity_tasks",
                    "balanced_performance",
                ],
            },
            {
                "name": "en_core_web_lg",
                "language": "English",
                "size": "large",
                "download_size_mb": 560,
                "description": "Large English model with high accuracy",
                "recommended_for": ["high_accuracy", "research", "comprehensive_nlp"],
            },
            {
                "name": "en_core_web_trf",
                "language": "English",
                "size": "transformer",
                "download_size_mb": 560,
                "description": "Transformer-based English model (highest accuracy)",
                "recommended_for": ["state_of_art", "research", "gpu_available"],
            },
        ]

        # Add validation status to each model
        for model_info in common_models:
            model_name = model_info["name"]
            model_info["cached"] = model_name in self._models
            if model_name in self._model_info:
                model_info["cache_info"] = self._model_info[model_name]

        return common_models

    def recommend_model(
        self, use_case: str = "general", resource_constraint: str = "medium"
    ) -> str:
        """Recommend a spaCy model based on use case and constraints.

        Args:
            use_case: Intended use case ("general", "ner", "similarity", "accuracy")
            resource_constraint: Resource constraint level ("low", "medium", "high")

        Returns:
            Recommended model name
        """
        recommendations = {
            ("general", "low"): "en_core_web_sm",
            ("general", "medium"): "en_core_web_md",
            ("general", "high"): "en_core_web_lg",
            ("ner", "low"): "en_core_web_sm",
            ("ner", "medium"): "en_core_web_md",
            ("ner", "high"): "en_core_web_trf",
            ("similarity", "low"): "en_core_web_md",  # Needs word vectors
            ("similarity", "medium"): "en_core_web_md",
            ("similarity", "high"): "en_core_web_lg",
            ("accuracy", "low"): "en_core_web_md",
            ("accuracy", "medium"): "en_core_web_lg",
            ("accuracy", "high"): "en_core_web_trf",
        }

        recommended = recommendations.get(
            (use_case, resource_constraint), "en_core_web_sm"
        )

        logger.info(
            f"Recommended spaCy model: {recommended}",
            extra={"use_case": use_case, "resource_constraint": resource_constraint},
        )

        return recommended

    def clear_cache(self):
        """Clear all cached models to free memory."""
        logger.info(f"Clearing {len(self._models)} cached spaCy models")

        for model_name in list(self._models.keys()):
            try:
                # spaCy models don't need explicit cleanup, just remove references
                del self._models[model_name]
                del self._model_info[model_name]
            except Exception as e:
                logger.warning(f"Error clearing spaCy model {model_name}: {e}")

        self._models.clear()
        self._model_info.clear()

        # Force garbage collection
        import gc

        gc.collect()

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about cached models.

        Returns:
            Dictionary with cache statistics and model information
        """
        return {
            "cached_models": list(self._models.keys()),
            "model_count": len(self._models),
            "model_details": dict(self._model_info),
        }

    def preload_models(self, model_names: list[str]) -> dict[str, bool]:
        """Preload multiple spaCy models for faster access.

        Args:
            model_names: List of model names to preload

        Returns:
            Dictionary mapping model names to load success status
        """
        results = {}

        for model_name in model_names:
            try:
                model = self.get_model(model_name)
                results[model_name] = model is not None
                if model is not None:
                    logger.info(f"Preloaded spaCy model: {model_name}")
                else:
                    logger.warning(f"Failed to preload spaCy model: {model_name}")
            except Exception as e:
                logger.error(f"Error preloading spaCy model {model_name}: {e}")
                results[model_name] = False

        success_count = sum(results.values())
        logger.info(f"Preloaded {success_count}/{len(model_names)} spaCy models")

        return results


# Global model manager instance
_spacy_manager: SpacyModelManager | None = None


def get_spacy_model_manager() -> SpacyModelManager:
    """Get or create the global spaCy model manager instance.

    Returns:
        Global SpacyModelManager for centralized model management

    Note:
        Uses singleton pattern to ensure consistent model management
        across the application.
    """
    global _spacy_manager
    if _spacy_manager is None:
        _spacy_manager = SpacyModelManager()
    return _spacy_manager


def validate_spacy_installation() -> dict[str, Any]:
    """Validate spaCy installation and available models.

    Returns:
        Dictionary with installation validation results
    """
    validation = {
        "spacy_installed": False,
        "spacy_version": None,
        "models_available": [],
        "installation_issues": [],
        "recommendations": [],
    }

    try:
        import spacy

        validation["spacy_installed"] = True
        validation["spacy_version"] = spacy.__version__

        # Check for common models
        common_models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
        for model_name in common_models:
            try:
                spacy.load(model_name)
                validation["models_available"].append(model_name)
            except OSError:
                # Model not installed
                pass

        if not validation["models_available"]:
            validation["installation_issues"].append("No spaCy models found")
            validation["recommendations"].append(
                "Install a spaCy model: python -m spacy download en_core_web_sm"
            )

        logger.info(
            f"spaCy validation: {validation['spacy_version']}, "
            f"{len(validation['models_available'])} models available"
        )

    except ImportError:
        validation["installation_issues"].append("spaCy not installed")
        validation["recommendations"].append("Install spaCy: pip install spacy")

    return validation


def get_model_download_command(model_name: str) -> str:
    """Get the command to download a specific spaCy model.

    Args:
        model_name: Name of the spaCy model

    Returns:
        Shell command to download the model
    """
    return f"python -m spacy download {model_name}"


def estimate_model_memory_usage(model_name: str) -> dict[str, Any]:
    """Estimate memory usage for a spaCy model.

    Args:
        model_name: Name of the spaCy model

    Returns:
        Dictionary with estimated memory usage information
    """
    # Estimated memory usage based on model size
    memory_estimates = {
        "en_core_web_sm": {"ram_mb": 20, "download_mb": 15},
        "en_core_web_md": {"ram_mb": 85, "download_mb": 40},
        "en_core_web_lg": {"ram_mb": 750, "download_mb": 560},
        "en_core_web_trf": {"ram_mb": 900, "download_mb": 560},
    }

    estimate = memory_estimates.get(model_name, {"ram_mb": 50, "download_mb": 25})
    estimate["model_name"] = model_name
    estimate["estimate_source"] = (
        "typical_usage" if model_name in memory_estimates else "fallback"
    )

    return estimate
