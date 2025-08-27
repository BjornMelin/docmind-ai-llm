#!/usr/bin/env python3
"""Validation integration tests that connect validation suite with existing validation scripts.

This module provides integration between the test validation framework and
the existing validation scripts in the scripts/ directory, creating a
comprehensive validation ecosystem.

Integration Areas:
- GPU validation (scripts/gpu_validation.py)
- Performance validation (scripts/performance_validation.py)
- End-to-end validation (scripts/end_to_end_test.py)
- System health monitoring and reporting
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from src.config.settings import DocMindSettings


class ValidationIntegrationRunner:
    """Runs validation scripts and collects results."""

    def __init__(self):
        self.results: dict[str, Any] = {}
        self.script_dir = Path(__file__).parent.parent.parent / "scripts"

    def run_script(self, script_name: str, timeout: int = 300) -> dict[str, Any]:
        """Run a validation script and capture results."""
        script_path = self.script_dir / script_name

        if not script_path.exists():
            return {
                "success": False,
                "error": f"Script not found: {script_path}",
                "stdout": "",
                "stderr": "",
                "duration_seconds": 0,
            }

        start_time = time.time()

        try:
            # Run the script with uv
            result = subprocess.run(
                ["uv", "run", "python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=script_path.parent.parent,
            )

            duration = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_seconds": duration,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Script timed out after {timeout}s",
                "stdout": "",
                "stderr": "",
                "duration_seconds": timeout,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run script: {e}",
                "stdout": "",
                "stderr": "",
                "duration_seconds": time.time() - start_time,
            }

    def extract_performance_metrics(self, stdout: str) -> dict[str, float]:
        """Extract performance metrics from validation script output."""
        metrics = {}

        # Common performance patterns to look for
        patterns = [
            ("GPU Memory Usage", r"GPU Memory: ([\d.]+)GB"),
            ("Load Time", r"Load time: ([\d.]+)ms"),
            ("Tokens per Second", r"Tokens/s: ([\d.]+)"),
            ("Latency", r"Latency: ([\d.]+)ms"),
            ("VRAM Usage", r"VRAM: ([\d.]+)GB"),
            ("CPU Usage", r"CPU: ([\d.]+)%"),
        ]

        import re

        for metric_name, pattern in patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                try:
                    metrics[metric_name.lower().replace(" ", "_")] = float(
                        match.group(1)
                    )
                except ValueError:
                    continue

        return metrics


@pytest.fixture(scope="session")
def validation_runner():
    """Session-scoped validation integration runner."""
    return ValidationIntegrationRunner()


@pytest.mark.integration
class TestGPUValidationIntegration:
    """Integration tests for GPU validation script."""

    def test_gpu_validation_script(self, validation_runner):
        """Test GPU validation script runs successfully."""
        result = validation_runner.run_script("gpu_validation.py", timeout=120)

        # Log results for debugging
        logging.info(f"GPU Validation Result: {result['success']}")
        if result["stdout"]:
            logging.info(f"GPU Validation Output: {result['stdout'][:500]}...")
        if result["stderr"]:
            logging.info(f"GPU Validation Errors: {result['stderr'][:500]}...")

        # Store results
        validation_runner.results["gpu_validation"] = result

        # GPU validation should complete successfully or give clear error
        if not result["success"] and "no GPU" not in result["stdout"].lower():
            pytest.fail(
                f"GPU validation failed unexpectedly: {result.get('error', 'Unknown error')}"
            )

    def test_gpu_validation_performance_metrics(self, validation_runner):
        """Test GPU validation provides performance metrics."""
        gpu_result = validation_runner.results.get("gpu_validation")

        if not gpu_result or not gpu_result["success"]:
            pytest.skip("GPU validation not successful, skipping metrics test")

        metrics = validation_runner.extract_performance_metrics(gpu_result["stdout"])

        # Should have some performance metrics
        assert len(metrics) > 0, "GPU validation should provide performance metrics"

        # Log metrics
        for metric, value in metrics.items():
            logging.info(f"GPU Metric - {metric}: {value}")


@pytest.mark.integration
class TestPerformanceValidationIntegration:
    """Integration tests for performance validation script."""

    def test_performance_validation_script(self, validation_runner):
        """Test performance validation script runs successfully."""
        result = validation_runner.run_script("performance_validation.py", timeout=180)

        # Log results
        logging.info(f"Performance Validation Result: {result['success']}")
        if result["stdout"]:
            logging.info(f"Performance Output: {result['stdout'][:500]}...")
        if result["stderr"]:
            logging.info(f"Performance Errors: {result['stderr'][:500]}...")

        # Store results
        validation_runner.results["performance_validation"] = result

        # Performance validation should provide meaningful output
        assert result["duration_seconds"] < 180, "Performance validation took too long"

        if not result["success"]:
            # Check if it's a known limitation
            if (
                "hardware" in result["stderr"].lower()
                or "gpu" in result["stderr"].lower()
            ):
                pytest.skip(
                    f"Performance validation skipped due to hardware: {result['stderr']}"
                )
            else:
                pytest.fail(
                    f"Performance validation failed: {result.get('error', 'Unknown error')}"
                )

    def test_performance_benchmarks(self, validation_runner):
        """Test performance validation provides benchmark data."""
        perf_result = validation_runner.results.get("performance_validation")

        if not perf_result or not perf_result["success"]:
            pytest.skip(
                "Performance validation not successful, skipping benchmark test"
            )

        metrics = validation_runner.extract_performance_metrics(perf_result["stdout"])

        # Should have performance benchmarks
        expected_metrics = ["load_time", "tokens_per_second", "latency", "vram_usage"]
        found_metrics = [
            m for m in expected_metrics if any(m in key for key in metrics.keys())
        ]

        assert len(found_metrics) > 0, (
            f"Expected performance metrics, got: {list(metrics.keys())}"
        )


@pytest.mark.integration
class TestEndToEndValidationIntegration:
    """Integration tests for end-to-end validation."""

    def test_end_to_end_validation_script(self, validation_runner):
        """Test end-to-end validation script runs successfully."""
        result = validation_runner.run_script("end_to_end_test.py", timeout=300)

        # Log results
        logging.info(f"E2E Validation Result: {result['success']}")
        if result["stdout"]:
            logging.info(f"E2E Output: {result['stdout'][:500]}...")
        if result["stderr"]:
            logging.info(f"E2E Errors: {result['stderr'][:500]}...")

        # Store results
        validation_runner.results["e2e_validation"] = result

        # End-to-end test is comprehensive, allow for some flexibility
        if not result["success"]:
            # Check for known issues
            error_msg = (result.get("error", "") + result.get("stderr", "")).lower()
            if any(
                issue in error_msg
                for issue in ["connection refused", "qdrant", "ollama"]
            ):
                pytest.skip(
                    f"E2E validation skipped due to external dependencies: {error_msg}"
                )
            else:
                pytest.fail(
                    f"E2E validation failed: {result.get('error', 'Unknown error')}"
                )


@pytest.mark.integration
class TestValidationReporting:
    """Test validation reporting and summary generation."""

    def test_validation_summary_generation(self, validation_runner):
        """Test comprehensive validation summary generation."""
        # Collect all validation results
        all_results = validation_runner.results

        # Generate summary
        summary = self._generate_validation_summary(all_results)

        # Summary should contain key information
        assert "validation_status" in summary
        assert "total_tests" in summary
        assert "successful_tests" in summary
        assert "failed_tests" in summary
        assert "performance_metrics" in summary

        # Log summary
        logging.info("=== VALIDATION INTEGRATION SUMMARY ===")
        logging.info(f"Status: {summary['validation_status']}")
        logging.info(
            f"Tests: {summary['successful_tests']}/{summary['total_tests']} successful"
        )

        if summary["performance_metrics"]:
            logging.info("Performance Metrics:")
            for metric, value in summary["performance_metrics"].items():
                logging.info(f"  {metric}: {value}")

        if summary["warnings"]:
            logging.info("Warnings:")
            for warning in summary["warnings"]:
                logging.info(f"  - {warning}")

        # Store summary for potential use by other tests
        validation_runner.summary = summary

    def test_validation_health_score(self, validation_runner):
        """Test calculation of overall validation health score."""
        if not hasattr(validation_runner, "summary"):
            pytest.skip("Validation summary not available")

        summary = validation_runner.summary

        # Calculate health score (0-100)
        total_tests = summary.get("total_tests", 0)
        successful_tests = summary.get("successful_tests", 0)

        if total_tests == 0:
            health_score = 0
        else:
            base_score = (successful_tests / total_tests) * 100

            # Adjust for warnings and performance
            warning_penalty = len(summary.get("warnings", [])) * 5
            health_score = max(0, base_score - warning_penalty)

        logging.info(f"Validation Health Score: {health_score:.1f}/100")

        # Health score should be reasonable
        assert 0 <= health_score <= 100, f"Invalid health score: {health_score}"

        # Store health score
        validation_runner.health_score = health_score

    def _generate_validation_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive validation summary."""
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r.get("success", False))
        failed_tests = total_tests - successful_tests

        # Aggregate performance metrics
        all_metrics = {}
        for test_name, result in results.items():
            if result.get("success") and result.get("stdout"):
                metrics = ValidationIntegrationRunner().extract_performance_metrics(
                    result["stdout"]
                )
                for metric, value in metrics.items():
                    all_metrics[f"{test_name}_{metric}"] = value

        # Collect warnings
        warnings = []
        for test_name, result in results.items():
            if not result.get("success"):
                error = result.get("error") or result.get("stderr", "Unknown error")
                if any(
                    skip_term in error.lower()
                    for skip_term in ["connection refused", "hardware", "skip"]
                ):
                    warnings.append(f"{test_name}: {error}")

        # Determine overall status
        if successful_tests == total_tests:
            status = "PASS"
        elif successful_tests >= total_tests * 0.7:  # 70% success rate
            status = "PARTIAL"
        else:
            status = "FAIL"

        return {
            "validation_status": status,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": (successful_tests / total_tests) * 100
            if total_tests > 0
            else 0,
            "performance_metrics": all_metrics,
            "warnings": warnings,
            "timestamp": time.time(),
        }


@pytest.mark.integration
class TestValidationEnvironmentCheck:
    """Test validation environment and dependencies."""

    def test_validation_environment_setup(self):
        """Test validation environment is properly set up."""
        # Check essential directories exist
        project_root = Path(__file__).parent.parent.parent
        essential_dirs = ["scripts", "src", "tests"]

        for dir_name in essential_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Essential directory missing: {dir_path}"

        # Check validation scripts exist
        scripts_dir = project_root / "scripts"
        validation_scripts = [
            "gpu_validation.py",
            "performance_validation.py",
            "end_to_end_test.py",
        ]

        for script in validation_scripts:
            script_path = scripts_dir / script
            if script_path.exists():
                logging.info(f"Found validation script: {script}")
            else:
                logging.warning(f"Missing validation script: {script}")

    def test_unified_config_integration(self):
        """Test integration with unified configuration system."""
        try:
            settings = DocMindSettings()

            # Verify unified config is accessible
            assert hasattr(settings, "vllm"), "Unified config missing vLLM settings"
            assert hasattr(settings, "agents"), "Unified config missing agents settings"
            assert hasattr(settings, "retrieval"), (
                "Unified config missing retrieval settings"
            )

            # Test config serialization for validation reporting
            config_dict = settings.model_dump()
            assert isinstance(config_dict, dict), "Config serialization failed"
            assert len(config_dict) > 0, "Config serialization returned empty dict"

            logging.info(
                f"Unified config loaded successfully with {len(config_dict)} settings"
            )

        except Exception as e:
            pytest.fail(f"Unified config integration failed: {e}")


if __name__ == "__main__":
    # Run validation integration tests
    pytest_args = [__file__, "-v", "--tb=short", "-m", "integration"]

    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)
