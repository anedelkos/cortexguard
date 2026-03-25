"""
Minimal Prometheus metrics module for CortexGuard Edge.

This module defines a small, stable set of metrics that provide
high‑value operational visibility without introducing cardinality risk.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram

# -----------------------------
# High‑value, low‑cardinality metrics
# -----------------------------

# 1. Duration of major subsystems (ms)
component_duration_ms = Histogram(
    "cortexguard_component_duration_ms",
    "Duration of major CortexGuard components in milliseconds",
    ["component"],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000),
)

# 2. Total anomalies emitted by the anomaly detector
anomalies_total = Counter(
    "cortexguard_anomalies_total",
    "Total number of anomalies emitted by the anomaly detector",
)

# 3. Sub‑detector failures
detector_failures_total = Counter(
    "cortexguard_detector_failures_total",
    "Total number of sub‑detector failures",
)

# 4. Policy escalations (Mayday)
policy_escalations_total = Counter(
    "cortexguard_policy_escalations_total",
    "Total number of remediation policies that required escalation",
)

# 5. Active anomalies (gauge)
active_anomalies = Gauge(
    "cortexguard_active_anomalies",
    "Number of currently active anomalies",
)

# 6. Plan queue length (gauge)
plan_queue_length = Gauge(
    "cortexguard_plan_queue_length",
    "Number of pending plans in the orchestrator queue",
)

# 7. Estimator confidence (gauge)
estimator_confidence = Gauge(
    "cortexguard_estimator_confidence",
    "Latest confidence value from the OnlineLearnerStateEstimator",
)

# HTTP ingestion
http_requests_total = Counter(
    "cortexguard_http_requests_total",
    "HTTP requests by outcome",
    ["method", "status_code"],
)
http_request_duration_ms = Histogram(
    "cortexguard_http_request_duration_ms",
    "HTTP ingestion latency (ms)",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000),
)

# LLM
llm_circuit_open = Gauge(
    "cortexguard_llm_circuit_open",
    "1 = circuit open (tripped), 0 = closed",
)
llm_requests_total = Counter(
    "cortexguard_llm_requests_total",
    "LLM requests by outcome",
    ["outcome"],  # success | failure | timeout | circuit_skipped
)

# Cloud escalation
mayday_escalations_total = Counter(
    "cortexguard_mayday_escalations_total",
    "Mayday attempts by outcome",
    ["outcome"],  # success | timeout | error | exhausted
)
mayday_consecutive_failures = Gauge(
    "cortexguard_mayday_consecutive_failures",
    "Current consecutive cloud escalation failures",
)

emergency_stop_active = Gauge(
    "cortexguard_emergency_stop_active",
    "1 if emergency stop is currently active, 0 otherwise",
)

# Plan and step execution
plans_total = Counter(
    "cortexguard_plans_total",
    "Plans by terminal status",
    ["status"],  # completed | failed
)
steps_total = Counter(
    "cortexguard_steps_total",
    "Steps by outcome",
    ["outcome"],  # completed | retry_exhausted | aborted
)


# -----------------------------
# Timing helper
# -----------------------------


@contextmanager
def record_duration(component: str) -> Iterator[None]:
    """
    Context manager to record duration of a component in ms.

    Usage:
        with record_duration("fusion"):
            fusion.process_record(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        component_duration_ms.labels(component=component).observe(elapsed_ms)
