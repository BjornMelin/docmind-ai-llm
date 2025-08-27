"""Performance Regression Detection System for DocMind AI.

This module provides automated performance regression detection with:
- Historical baseline management
- Threshold-based regression alerts
- Performance trend analysis
- Component-specific monitoring
- CI/CD integration support

Usage:
    from tests.performance.performance_regression_tracker import RegressionTracker

    tracker = RegressionTracker()
    tracker.record_performance("embedding_latency", 35.2, "ms")
    regression_detected = tracker.check_regression("embedding_latency")
"""

import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default regression thresholds
DEFAULT_THRESHOLDS = {
    "latency_regression_factor": 1.5,  # 50% increase triggers alert
    "memory_regression_mb": 200,  # 200MB increase triggers alert
    "throughput_degradation_factor": 0.7,  # 30% decrease triggers alert
    "min_samples_required": 3,  # Minimum samples needed for regression detection
}

# Performance baseline storage location
BASELINE_STORAGE_PATH = Path("tests/performance/baselines")


class PerformanceBaseline:
    """Represents a performance baseline for a specific metric."""

    def __init__(self, metric_name: str, metric_type: str = "latency"):
        """Initialize performance baseline.

        Args:
            metric_name: Name of the performance metric
            metric_type: Type of metric (latency, memory, throughput)
        """
        self.metric_name = metric_name
        self.metric_type = metric_type
        self.measurements: list[dict[str, Any]] = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

    def add_measurement(
        self, value: float, unit: str, metadata: dict | None = None
    ) -> None:
        """Add a performance measurement to the baseline.

        Args:
            value: Performance measurement value
            unit: Unit of measurement (ms, MB, rps, etc.)
            metadata: Optional metadata about the measurement
        """
        measurement = {
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.measurements.append(measurement)
        self.last_updated = datetime.now()

        # Keep only last 100 measurements for efficiency
        if len(self.measurements) > 100:
            self.measurements = self.measurements[-100:]

    def get_statistics(self, days_back: int = 30) -> dict[str, float]:
        """Get statistical summary of recent measurements.

        Args:
            days_back: Number of days to include in statistics

        Returns:
            Dictionary with statistical metrics (mean, p95, std, etc.)
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_measurements = [
            m["value"]
            for m in self.measurements
            if datetime.fromisoformat(m["timestamp"]) >= cutoff_date
        ]

        if not recent_measurements:
            return {"count": 0}

        return {
            "count": len(recent_measurements),
            "mean": statistics.mean(recent_measurements),
            "median": statistics.median(recent_measurements),
            "std": statistics.stdev(recent_measurements)
            if len(recent_measurements) > 1
            else 0.0,
            "min": min(recent_measurements),
            "max": max(recent_measurements),
            "p95": np.percentile(recent_measurements, 95),
            "p99": np.percentile(recent_measurements, 99),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert baseline to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type,
            "measurements": self.measurements,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceBaseline":
        """Create baseline from dictionary."""
        baseline = cls(data["metric_name"], data["metric_type"])
        baseline.measurements = data["measurements"]
        baseline.created_at = datetime.fromisoformat(data["created_at"])
        baseline.last_updated = datetime.fromisoformat(data["last_updated"])
        return baseline


class RegressionTracker:
    """Performance regression tracking and detection system."""

    def __init__(
        self, storage_path: Path | None = None, thresholds: dict | None = None
    ):
        """Initialize regression tracker.

        Args:
            storage_path: Path to store performance baselines
            thresholds: Custom regression detection thresholds
        """
        self.storage_path = storage_path or BASELINE_STORAGE_PATH
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.baselines: dict[str, PerformanceBaseline] = {}

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing baselines
        self._load_baselines()

    def record_performance(
        self,
        metric_name: str,
        value: float,
        unit: str,
        metric_type: str = "latency",
        metadata: dict | None = None,
    ) -> None:
        """Record a performance measurement.

        Args:
            metric_name: Name of the performance metric
            value: Performance measurement value
            unit: Unit of measurement
            metric_type: Type of metric (latency, memory, throughput)
            metadata: Optional metadata about the measurement
        """
        if metric_name not in self.baselines:
            self.baselines[metric_name] = PerformanceBaseline(metric_name, metric_type)

        self.baselines[metric_name].add_measurement(value, unit, metadata)

        # Save updated baseline
        self._save_baseline(metric_name)

        logger.debug(f"Recorded {metric_name}: {value} {unit}")

    def check_regression(
        self,
        metric_name: str,
        current_value: float | None = None,
        days_back: int = 30,
    ) -> dict[str, Any]:
        """Check for performance regression in a metric.

        Args:
            metric_name: Name of the metric to check
            current_value: Optional current value to compare against baseline
            days_back: Number of days to include in baseline calculation

        Returns:
            Dictionary with regression analysis results
        """
        if metric_name not in self.baselines:
            return {
                "regression_detected": False,
                "error": f"No baseline found for metric: {metric_name}",
            }

        baseline = self.baselines[metric_name]
        stats = baseline.get_statistics(days_back)

        if stats["count"] < self.thresholds["min_samples_required"]:
            return {
                "regression_detected": False,
                "warning": f"Insufficient samples ({stats['count']}) for regression detection",
            }

        # Use current value or latest measurement
        if current_value is None:
            if not baseline.measurements:
                return {
                    "regression_detected": False,
                    "error": "No measurements available",
                }
            current_value = baseline.measurements[-1]["value"]

        # Determine regression based on metric type
        regression_detected = False
        regression_factor = 1.0

        if baseline.metric_type == "latency":
            # Higher latency is regression
            threshold = stats["p95"] * self.thresholds["latency_regression_factor"]
            regression_detected = current_value > threshold
            regression_factor = (
                current_value / stats["p95"] if stats["p95"] > 0 else 1.0
            )

        elif baseline.metric_type == "memory":
            # Higher memory usage is regression
            threshold = stats["mean"] + self.thresholds["memory_regression_mb"]
            regression_detected = current_value > threshold
            regression_factor = (
                (current_value - stats["mean"]) / stats["mean"]
                if stats["mean"] > 0
                else 1.0
            )

        elif baseline.metric_type == "throughput":
            # Lower throughput is regression
            threshold = stats["mean"] * self.thresholds["throughput_degradation_factor"]
            regression_detected = current_value < threshold
            regression_factor = (
                current_value / stats["mean"] if stats["mean"] > 0 else 1.0
            )

        return {
            "regression_detected": regression_detected,
            "regression_factor": regression_factor,
            "current_value": current_value,
            "baseline_stats": stats,
            "threshold_used": threshold if "threshold" in locals() else None,
            "metric_type": baseline.metric_type,
            "analysis_date": datetime.now().isoformat(),
        }

    def get_trend_analysis(
        self, metric_name: str, days_back: int = 90
    ) -> dict[str, Any]:
        """Analyze performance trends for a metric.

        Args:
            metric_name: Name of the metric to analyze
            days_back: Number of days to include in analysis

        Returns:
            Dictionary with trend analysis results
        """
        if metric_name not in self.baselines:
            return {"error": f"No baseline found for metric: {metric_name}"}

        baseline = self.baselines[metric_name]
        cutoff_date = datetime.now() - timedelta(days=days_back)

        # Filter recent measurements
        recent_measurements = [
            m
            for m in baseline.measurements
            if datetime.fromisoformat(m["timestamp"]) >= cutoff_date
        ]

        if len(recent_measurements) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Calculate trend (linear regression slope)
        timestamps = [
            datetime.fromisoformat(m["timestamp"]).timestamp()
            for m in recent_measurements
        ]
        values = [m["value"] for m in recent_measurements]

        # Normalize timestamps to start from 0
        timestamps = [t - min(timestamps) for t in timestamps]

        # Simple linear regression
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values, strict=False))
        sum_x2 = sum(x * x for x in timestamps)

        slope = (
            (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            if n * sum_x2 != sum_x * sum_x
            else 0
        )

        # Determine trend direction
        trend_direction = (
            "improving"
            if slope < 0 and baseline.metric_type != "throughput"
            else "improving"
            if slope > 0 and baseline.metric_type == "throughput"
            else "degrading"
            if slope != 0
            else "stable"
        )

        return {
            "metric_name": metric_name,
            "trend_direction": trend_direction,
            "trend_slope": slope,
            "data_points": len(recent_measurements),
            "date_range": {
                "start": recent_measurements[0]["timestamp"],
                "end": recent_measurements[-1]["timestamp"],
            },
            "value_range": {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
            },
        }

    def generate_regression_report(self, days_back: int = 7) -> dict[str, Any]:
        """Generate comprehensive regression report for all metrics.

        Args:
            days_back: Number of days to include in analysis

        Returns:
            Dictionary with regression report
        """
        report = {
            "report_date": datetime.now().isoformat(),
            "analysis_period_days": days_back,
            "metrics_analyzed": len(self.baselines),
            "regressions_detected": [],
            "improvements_detected": [],
            "stable_metrics": [],
            "insufficient_data": [],
        }

        for metric_name in self.baselines.keys():
            regression_check = self.check_regression(metric_name, days_back=days_back)

            if "error" in regression_check or "warning" in regression_check:
                report["insufficient_data"].append(
                    {
                        "metric": metric_name,
                        "reason": regression_check.get("error")
                        or regression_check.get("warning"),
                    }
                )
            elif regression_check["regression_detected"]:
                report["regressions_detected"].append(
                    {
                        "metric": metric_name,
                        "factor": regression_check["regression_factor"],
                        "current_value": regression_check["current_value"],
                        "baseline_p95": regression_check["baseline_stats"]["p95"],
                    }
                )
            elif regression_check["regression_factor"] < 0.8:  # Significant improvement
                report["improvements_detected"].append(
                    {
                        "metric": metric_name,
                        "factor": regression_check["regression_factor"],
                        "current_value": regression_check["current_value"],
                    }
                )
            else:
                report["stable_metrics"].append(metric_name)

        return report

    def _load_baselines(self) -> None:
        """Load performance baselines from storage."""
        try:
            for baseline_file in self.storage_path.glob("*.json"):
                with open(baseline_file) as f:
                    data = json.load(f)
                    baseline = PerformanceBaseline.from_dict(data)
                    self.baselines[baseline.metric_name] = baseline

            logger.info(f"Loaded {len(self.baselines)} performance baselines")

        except Exception as e:
            logger.warning(f"Failed to load baselines: {e}")

    def _save_baseline(self, metric_name: str) -> None:
        """Save a performance baseline to storage."""
        try:
            baseline_file = self.storage_path / f"{metric_name}.json"
            with open(baseline_file, "w") as f:
                json.dump(self.baselines[metric_name].to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save baseline for {metric_name}: {e}")

    def cleanup_old_data(self, days_to_keep: int = 90) -> None:
        """Clean up old performance data beyond retention period.

        Args:
            days_to_keep: Number of days of data to retain
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        for baseline in self.baselines.values():
            original_count = len(baseline.measurements)
            baseline.measurements = [
                m
                for m in baseline.measurements
                if datetime.fromisoformat(m["timestamp"]) >= cutoff_date
            ]
            cleaned_count = original_count - len(baseline.measurements)

            if cleaned_count > 0:
                logger.info(
                    f"Cleaned {cleaned_count} old measurements from {baseline.metric_name}"
                )
                self._save_baseline(baseline.metric_name)


# Convenience functions for direct usage
def record_latency(
    metric_name: str, latency_ms: float, metadata: dict | None = None
) -> None:
    """Record a latency measurement.

    Args:
        metric_name: Name of the latency metric
        latency_ms: Latency in milliseconds
        metadata: Optional metadata
    """
    tracker = RegressionTracker()
    tracker.record_performance(metric_name, latency_ms, "ms", "latency", metadata)


def record_throughput(
    metric_name: str, rps: float, metadata: dict | None = None
) -> None:
    """Record a throughput measurement.

    Args:
        metric_name: Name of the throughput metric
        rps: Requests per second
        metadata: Optional metadata
    """
    tracker = RegressionTracker()
    tracker.record_performance(metric_name, rps, "rps", "throughput", metadata)


def record_memory(
    metric_name: str, memory_mb: float, metadata: dict | None = None
) -> None:
    """Record a memory usage measurement.

    Args:
        metric_name: Name of the memory metric
        memory_mb: Memory usage in MB
        metadata: Optional metadata
    """
    tracker = RegressionTracker()
    tracker.record_performance(metric_name, memory_mb, "MB", "memory", metadata)


def check_for_regressions(metrics: list[str] | None = None) -> dict[str, Any]:
    """Check for regressions in specified metrics or all metrics.

    Args:
        metrics: List of metric names to check, or None for all metrics

    Returns:
        Dictionary with regression analysis results
    """
    tracker = RegressionTracker()

    if metrics is None:
        return tracker.generate_regression_report()

    results = {}
    for metric in metrics:
        results[metric] = tracker.check_regression(metric)

    return results
