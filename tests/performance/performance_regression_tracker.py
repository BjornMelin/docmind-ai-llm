"""Minimal performance regression tracker used by scripts/performance_monitor.py.

Persists a simple JSON baseline of key metrics and compares current values
against the baseline using a percentage threshold.

Baseline path: tests/performance/baselines/baseline.json
Current data: provided by caller (preferred). If unavailable, attempts to use
the most recent report file in tests/performance/reports/ (optional).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASELINE_DIR = Path("tests/performance/baselines")
BASELINE_FILE = BASELINE_DIR / "baseline.json"
REPORTS_DIR = Path("tests/performance/reports")


class RegressionTracker:
    """Simple baseline store + regression checker."""

    def __init__(self) -> None:
        """Initialize tracker."""
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Baseline IO ---
    def _load_baseline(self) -> dict[str, Any]:
        if BASELINE_FILE.exists():
            with BASELINE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_baseline(self, data: dict[str, Any]) -> None:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        with BASELINE_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # --- Public API used by performance_monitor ---
    def record_performance(
        self,
        metric: str,
        value: float,
        unit: str,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update baseline with a single metric value.

        Called by PerformanceMonitor.record_performance_baseline().
        """
        baseline = self._load_baseline()
        baseline.setdefault("metrics", {})
        baseline["metrics"][metric] = {
            "value": float(value),
            "unit": unit,
            "kind": kind,
            "metadata": metadata or {},
        }
        self._save_baseline(baseline)

    def _extract_current_value(
        self, metric: str, current_data: dict[str, Any] | None
    ) -> float | None:
        """Best-effort extraction of current value for a known metric name."""
        if not current_data:
            return None
        # Map our canonical metric names to PerformanceMonitor result keys
        mapping = {
            "test_suite_duration": ["total_duration", "duration", "suite_duration"],
            "test_collection_time": ["collection_time"],
            "average_test_duration": ["average_duration"],
            # The following are not currently emitted by the monitor;
            # keep for future use
            "memory_usage_peak": ["memory_usage_peak"],
            "gpu_vram_peak_mb": ["gpu_vram_peak_mb"],
            "embedding_latency": ["embedding_latency"],
            "retrieval_latency": ["retrieval_latency"],
            "llm_inference_time": ["llm_inference_time"],
        }
        keys = mapping.get(metric, [])
        for k in keys:
            if k in current_data:
                try:
                    return float(current_data[k])
                except (TypeError, ValueError):
                    return None
        return None

    def _latest_report_value(self, metric: str) -> float | None:
        """Fallback: inspect the latest saved report for the metric."""
        reports = sorted(REPORTS_DIR.glob("performance_data_*.json"), reverse=True)
        for p in reports:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                val = self._extract_current_value(metric, data)
                if val is not None:
                    return val
            except (OSError, ValueError, UnicodeDecodeError):
                continue
        return None

    def check_regression(
        self,
        metric: str,
        *,
        current_data: dict[str, Any] | None = None,
        threshold_pct: float = 20.0,
    ) -> dict[str, Any]:
        """Compare current metric value against baseline.

        Returns a dictionary with:
          - regression_detected (bool)
          - baseline_value, current_value, regression_factor, threshold_pct
          - reason (optional diagnostic)
        """
        baseline = self._load_baseline()
        base_metrics = baseline.get("metrics", {})
        base_entry = base_metrics.get(metric)
        if not base_entry:
            return {"regression_detected": False, "reason": "no_baseline"}

        baseline_value = float(base_entry.get("value", 0.0))
        if baseline_value <= 0:
            return {"regression_detected": False, "reason": "invalid_baseline"}

        current_value = self._extract_current_value(metric, current_data)
        if current_value is None:
            current_value = self._latest_report_value(metric)
        if current_value is None:
            return {"regression_detected": False, "reason": "no_current_data"}

        # Compute percentage increase (regression when larger-is-worse)
        delta = current_value - baseline_value
        pct_increase = (delta / baseline_value) * 100.0
        regression = pct_increase > threshold_pct
        factor = (current_value / baseline_value) if baseline_value else 1.0

        return {
            "regression_detected": bool(regression),
            "baseline_value": baseline_value,
            "current_value": current_value,
            "regression_factor": factor,
            "threshold_pct": threshold_pct,
            "increase_pct": pct_increase,
        }

    def get_trend_analysis(self, metric: str, days_back: int = 30) -> dict[str, Any]:
        """Simple stub trend analysis; returns a stable trend with zero samples.

        A future implementation can persist daily snapshots and compute a real trend.
        """
        return {"trend_direction": "stable", "data_points": 0}
