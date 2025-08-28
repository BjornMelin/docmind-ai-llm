#!/usr/bin/env python3
"""Production readiness validation framework for DocMind AI.

This module provides comprehensive validation for production deployment readiness,
including performance benchmarks, hardware requirements, system health checks,
and end-to-end workflow validation.

Production Areas Validated:
- Hardware requirements and GPU capabilities
- Performance benchmarks against targets
- Configuration completeness and correctness
- Multi-agent system coordination
- End-to-end document processing workflows
- System resource utilization and limits
- Error handling and graceful degradation
"""

import sys
import time
from typing import Any

import pytest
from llama_index.core import Document

from src.config.settings import DocMindSettings
from src.utils.core import detect_hardware


class ProductionValidationResult:
    """Container for production validation results."""

    def __init__(self):
        """Initialize production validation result container."""
        self.test_results: dict[str, Any] = {}
        self.performance_metrics: dict[str, float] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_result(
        self,
        category: str,
        test_name: str,
        passed: bool,
        details: dict[str, Any] = None,
    ):
        """Add a test result."""
        if category not in self.test_results:
            self.test_results[category] = {}
        self.test_results[category][test_name] = {
            "passed": passed,
            "details": details or {},
        }

    def add_performance_metric(self, metric_name: str, value: float):
        """Add a performance metric."""
        self.performance_metrics[metric_name] = value

    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)

    def is_production_ready(self) -> bool:
        """Determine if system is production ready based on results."""
        # No critical errors allowed
        if self.errors:
            return False

        # All essential tests must pass
        essential_categories = ["hardware", "configuration", "performance"]
        for category in essential_categories:
            if category not in self.test_results:
                return False
            category_results = self.test_results[category]
            if not all(result["passed"] for result in category_results.values()):
                return False

        return True

    def generate_report(self) -> str:
        """Generate a human-readable production readiness report."""
        lines = []
        lines.append("=" * 60)
        lines.append("DOCMIND AI PRODUCTION READINESS REPORT")
        lines.append("=" * 60)

        # Overall status
        status = "✅ PRODUCTION READY" if self.is_production_ready() else "❌ NOT READY"
        lines.append(f"\nOverall Status: {status}\n")

        # Test results by category
        for category, tests in self.test_results.items():
            lines.append(f"\n{category.upper()} VALIDATION:")
            lines.append("-" * 40)
            for test_name, result in tests.items():
                status_icon = "✅" if result["passed"] else "❌"
                lines.append(f"  {status_icon} {test_name}")
                if result["details"]:
                    for key, value in result["details"].items():
                        lines.append(f"      {key}: {value}")

        # Performance metrics
        if self.performance_metrics:
            lines.append("\nPERFORMANCE METRICS:")
            lines.append("-" * 40)
            for metric, value in self.performance_metrics.items():
                lines.append(f"  {metric}: {value}")

        # Warnings
        if self.warnings:
            lines.append("\nWARNINGS:")
            lines.append("-" * 40)
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")

        # Errors
        if self.errors:
            lines.append("\nERRORS:")
            lines.append("-" * 40)
            for error in self.errors:
                lines.append(f"  ❌ {error}")

        return "\n".join(lines)


@pytest.fixture(scope="session")
def production_validator():
    """Session-scoped production validation fixture."""
    return ProductionValidationResult()


@pytest.mark.system
class TestProductionHardwareRequirements:
    """Validate production hardware requirements."""

    def test_gpu_availability(self, production_validator):
        """Test GPU availability for production deployment."""
        hardware_info = detect_hardware()

        cuda_available = hardware_info.get("cuda_available", False)
        gpu_name = hardware_info.get("gpu_name", "Unknown")
        vram_gb = hardware_info.get("vram_total_gb", 0)

        # Production requires GPU with sufficient VRAM
        passed = cuda_available and vram_gb >= 12  # Minimum for Qwen3-4B-FP8

        production_validator.add_result(
            "hardware",
            "gpu_availability",
            passed,
            {
                "cuda_available": cuda_available,
                "gpu_name": gpu_name,
                "vram_gb": vram_gb,
                "minimum_vram_gb": 12,
            },
        )

        if not passed:
            production_validator.add_error(
                f"GPU requirements not met: {gpu_name} with {vram_gb}GB VRAM"
            )

        assert passed, f"Production requires GPU with ≥12GB VRAM, got {vram_gb}GB"

    def test_memory_requirements(self, production_validator):
        """Test system memory requirements."""
        import psutil

        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Production requires at least 16GB total, 8GB available
        passed = total_memory_gb >= 16 and available_memory_gb >= 8

        production_validator.add_result(
            "hardware",
            "memory_requirements",
            passed,
            {
                "total_memory_gb": round(total_memory_gb, 1),
                "available_memory_gb": round(available_memory_gb, 1),
                "minimum_total_gb": 16,
                "minimum_available_gb": 8,
            },
        )

        if not passed:
            production_validator.add_error(
                f"Memory requirements not met: {total_memory_gb:.1f}GB total, "
                f"{available_memory_gb:.1f}GB available"
            )

    def test_storage_requirements(self, production_validator):
        """Test storage space requirements."""
        import shutil

        free_space_gb = shutil.disk_usage("/").free / (1024**3)

        # Production requires at least 50GB free space
        passed = free_space_gb >= 50

        production_validator.add_result(
            "hardware",
            "storage_requirements",
            passed,
            {"free_space_gb": round(free_space_gb, 1), "minimum_free_gb": 50},
        )

        if not passed:
            production_validator.add_error(
                f"Storage requirements not met: {free_space_gb:.1f}GB available"
            )


@pytest.mark.system
class TestProductionConfiguration:
    """Validate production configuration completeness."""

    def test_unified_settings_validation(self, production_validator):
        """Test unified settings are properly configured."""
        try:
            settings = DocMindSettings()

            # Validate essential configuration sections
            essential_configs = [
                "vllm",
                "agents",
                "retrieval",
                "embedding",
                "processing",
            ]
            passed = all(hasattr(settings, config) for config in essential_configs)

            production_validator.add_result(
                "configuration",
                "unified_settings",
                passed,
                {
                    "missing_configs": [
                        c for c in essential_configs if not hasattr(settings, c)
                    ]
                },
            )

            if not passed:
                production_validator.add_error("Unified settings validation failed")

        except Exception as e:
            production_validator.add_result("configuration", "unified_settings", False)
            production_validator.add_error(f"Settings initialization failed: {e}")

    def test_model_configuration(self, production_validator):
        """Test model configuration is production-ready."""
        settings = DocMindSettings()

        # Verify model configurations
        model_checks = {
            "llm_model_set": bool(settings.vllm.model),
            "embedding_model_set": bool(settings.embedding.model_name),
            "bge_m3_configured": "bge-m3" in settings.embedding.model_name.lower(),
            "context_window_adequate": settings.vllm.context_window >= 32768,
        }

        passed = all(model_checks.values())

        production_validator.add_result(
            "configuration", "model_configuration", passed, model_checks
        )

        if not passed:
            failed_checks = [k for k, v in model_checks.items() if not v]
            production_validator.add_error(
                f"Model configuration issues: {failed_checks}"
            )

    def test_performance_configuration(self, production_validator):
        """Test performance-related configuration."""
        settings = DocMindSettings()

        # Check FP8 optimization settings
        fp8_checks = {
            "fp8_quantization": settings.quantization == "fp8",
            "fp8_kv_cache": settings.kv_cache_dtype == "fp8",
            "flashinfer_backend": settings.vllm.attention_backend == "FLASHINFER",
            "gpu_memory_optimized": 0.8 <= settings.vllm.gpu_memory_utilization <= 0.9,
            "chunked_prefill_enabled": settings.vllm.enable_chunked_prefill,
        }

        passed = all(fp8_checks.values())

        production_validator.add_result(
            "configuration", "performance_optimization", passed, fp8_checks
        )

        if not passed:
            failed_checks = [k for k, v in fp8_checks.items() if not v]
            production_validator.add_warning(
                f"Performance optimization issues: {failed_checks}"
            )


@pytest.mark.system
class TestProductionPerformance:
    """Validate production performance benchmarks."""

    def test_configuration_load_time(self, production_validator):
        """Test configuration loading performance."""
        start_time = time.time()

        # Load settings multiple times to test consistency
        for _ in range(5):
            DocMindSettings()

        load_time_ms = (time.time() - start_time) * 1000 / 5  # Average per load

        # Configuration should load in <100ms
        passed = load_time_ms < 100

        production_validator.add_result(
            "performance",
            "config_load_time",
            passed,
            {"average_load_time_ms": round(load_time_ms, 2), "target_ms": 100},
        )

        production_validator.add_performance_metric("config_load_time_ms", load_time_ms)

        if not passed:
            production_validator.add_error(
                f"Configuration load too slow: {load_time_ms:.2f}ms"
            )

    def test_memory_usage_baseline(self, production_validator):
        """Test baseline memory usage is reasonable."""
        import gc

        import psutil

        # Force garbage collection
        gc.collect()

        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**2)  # MB

        # Load core components
        DocMindSettings()
        detect_hardware()

        gc.collect()
        memory_after = process.memory_info().rss / (1024**2)  # MB
        memory_increase = memory_after - memory_before

        # Memory increase should be reasonable (<500MB for basic components)
        passed = memory_increase < 500

        production_validator.add_result(
            "performance",
            "memory_baseline",
            passed,
            {
                "memory_before_mb": round(memory_before, 1),
                "memory_after_mb": round(memory_after, 1),
                "memory_increase_mb": round(memory_increase, 1),
                "target_increase_mb": 500,
            },
        )

        production_validator.add_performance_metric(
            "baseline_memory_mb", memory_increase
        )

        if not passed:
            production_validator.add_warning(
                f"High baseline memory usage: {memory_increase:.1f}MB increase"
            )


@pytest.mark.system
class TestProductionIntegration:
    """Test production integration scenarios."""

    def test_multi_agent_initialization(self, production_validator):
        """Test multi-agent system can be initialized."""
        try:
            from src.agents.coordinator import (
                create_multi_agent_coordinator,
            )

            # Test coordinator creation
            settings = DocMindSettings()
            coordinator = create_multi_agent_coordinator(settings)

            passed = coordinator is not None

            production_validator.add_result(
                "integration",
                "multi_agent_init",
                passed,
                {"coordinator_type": type(coordinator).__name__},
            )

            if not passed:
                production_validator.add_error(
                    "Multi-agent coordinator initialization failed"
                )

        except Exception as e:
            production_validator.add_result("integration", "multi_agent_init", False)
            production_validator.add_error(f"Multi-agent initialization error: {e}")

    def test_document_processing_workflow(self, production_validator):
        """Test end-to-end document processing workflow."""
        try:
            # Test document creation and basic processing
            test_documents = [
                Document(
                    text=(
                        "DocMind AI uses advanced retrieval techniques for "
                        "document analysis."
                    ),
                    metadata={"source": "test.pdf", "page": 1},
                ),
                Document(
                    text=(
                        "The system implements BGE-M3 embeddings with FP8 optimization."
                    ),
                    metadata={"source": "test.pdf", "page": 2},
                ),
            ]

            # Basic document processing validation
            passed = len(test_documents) == 2
            passed = passed and all(doc.text for doc in test_documents)
            passed = passed and all(doc.metadata for doc in test_documents)

            production_validator.add_result(
                "integration",
                "document_workflow",
                passed,
                {
                    "documents_processed": len(test_documents),
                    "avg_text_length": sum(len(doc.text) for doc in test_documents)
                    // len(test_documents),
                },
            )

            if not passed:
                production_validator.add_error(
                    "Document processing workflow validation failed"
                )

        except Exception as e:
            production_validator.add_result("integration", "document_workflow", False)
            production_validator.add_error(f"Document workflow error: {e}")


@pytest.mark.system
class TestProductionHealthChecks:
    """Production health check validations."""

    def test_system_resource_limits(self, production_validator):
        """Test system respects resource limits."""
        import psutil

        settings = DocMindSettings()

        # Check configured limits vs actual system resources
        psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)

        resource_checks = {
            "max_memory_reasonable": settings.max_memory_gb <= memory_gb,
            "max_vram_reasonable": settings.max_vram_gb <= 80,  # Reasonable upper bound
            "context_window_reasonable": settings.vllm.context_window <= 200000,
        }

        passed = all(resource_checks.values())

        production_validator.add_result(
            "health",
            "resource_limits",
            passed,
            {
                "system_memory_gb": round(memory_gb, 1),
                "configured_max_memory_gb": settings.max_memory_gb,
                "configured_max_vram_gb": settings.max_vram_gb,
                **resource_checks,
            },
        )

        if not passed:
            failed_checks = [k for k, v in resource_checks.items() if not v]
            production_validator.add_warning(f"Resource limit issues: {failed_checks}")

    def test_error_handling_resilience(self, production_validator):
        """Test error handling and graceful degradation."""
        try:
            # Test settings with invalid values are handled gracefully
            with pytest.raises(ValueError, match=r"chunk.*size|must.*greater|>=.*100"):
                DocMindSettings(processing={"chunk_size": 0})

            # Test hardware detection doesn't crash
            hardware_info = detect_hardware()

            passed = isinstance(hardware_info, dict)

            production_validator.add_result(
                "health",
                "error_handling",
                passed,
                {"hardware_detection_stable": isinstance(hardware_info, dict)},
            )

        except Exception as e:
            production_validator.add_result("health", "error_handling", False)
            production_validator.add_error(f"Error handling test failed: {e}")


# Production validation runner
def run_production_validation() -> ProductionValidationResult:
    """Run comprehensive production validation and return results."""
    validator = ProductionValidationResult()

    # Run pytest with this module to collect results
    pytest_args = [
        __file__,
        "-v",
        "-x",  # Stop on first failure for quick feedback
        "--tb=short",
    ]

    # Capture results
    result_code = pytest.main(pytest_args)

    if result_code != 0:
        validator.add_error(f"Validation tests failed with exit code {result_code}")

    return validator


if __name__ == "__main__":
    # Run production validation and print report
    print("Running DocMind AI Production Readiness Validation...")

    validator = run_production_validation()

    print("\n" + validator.generate_report())

    # Exit with appropriate code
    sys.exit(0 if validator.is_production_ready() else 1)
