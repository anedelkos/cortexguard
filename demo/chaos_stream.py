"""Chaos scenario streaming demo.

Streams a named anomaly scenario to a running CortexGuard edge service
and prints live feedback from the runtime-metrics endpoint.

Usage::

    python demo/chaos_stream.py --list
    python demo/chaos_stream.py --scenario S0.1
    python demo/chaos_stream.py --scenario S1.1 --endpoint http://localhost:8080/api/v1/ingest --rate 2.0

Requires ``PYTHONPATH=src`` or run via ``task demo:scenario``.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

# Allow running directly as a script without setting PYTHONPATH externally.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cortexguard.simulation.chaos_engine import ChaosEngine
from cortexguard.simulation.models.windowed_fused_record import SensorReading, WindowedFusedRecord
from cortexguard.simulation.scenario_loader import Scenario, load_scenarios

# ---------------------------------------------------------------------------
# HTTP helper — prefer httpx (transitive dep), fall back to stdlib
# ---------------------------------------------------------------------------
try:
    import httpx as _httpx

    def _post_json(url: str, payload: dict[str, Any]) -> int:
        """POST JSON payload; return HTTP status code."""
        with _httpx.Client(timeout=10.0) as client:
            r = client.post(url, json=payload)
            return int(r.status_code)

    def _get_json(url: str) -> dict[str, Any] | None:
        """GET JSON from URL; return parsed dict or None on error."""
        try:
            with _httpx.Client(timeout=5.0) as client:
                r = client.get(url)
                if r.status_code == 200:
                    result: dict[str, Any] = r.json()
                    return result
        except Exception:
            pass
        return None

except ImportError:
    import json
    import urllib.request

    def _post_json(url: str, payload: dict[str, Any]) -> int:  # type: ignore[misc]
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return int(resp.status)

    def _get_json(url: str) -> dict[str, Any] | None:  # type: ignore[misc]
        import json as _json

        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return _json.loads(resp.read())  # type: ignore[no-any-return]
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Baseline record generator (mirrors make_synthetic_baseline in integration tests)
# ---------------------------------------------------------------------------


def _make_synthetic_baseline(
    n_records: int = 20,
    window_size: int = 5,
    dt_ns: int = 10_000_000,
    seed: int = 42,
) -> list[WindowedFusedRecord]:
    """Generate synthetic near-zero baseline records in memory.

    Produces the same deterministic stream as ``make_synthetic_baseline`` in
    ``tests/integration/test_anomaly_scenarios.py``.

    Args:
        n_records: Number of windowed records to generate.
        window_size: Sensor samples per window.
        dt_ns: Nanoseconds between record timestamps.
        seed: Random seed for reproducibility.

    Returns:
        List of WindowedFusedRecord instances.
    """
    rng = random.Random(seed)
    base_ts = 1_700_000_000_000_000_000
    records: list[WindowedFusedRecord] = []

    drift = {
        k: 0.0
        for k in [
            "force_x",
            "force_y",
            "force_z",
            "torque_x",
            "torque_y",
            "torque_z",
            "pos_x",
            "pos_y",
            "pos_z",
            "temp_c",
            "smoke_ppm",
        ]
    }

    for i in range(n_records):
        ts = base_ts + i * dt_ns

        for k in drift:
            drift[k] += rng.uniform(-0.005, 0.005)

        sensor_window = []
        for j in range(window_size):
            sensor_window.append(
                SensorReading(
                    timestamp_ns=ts - (window_size - j) * 1_000_000,
                    force_x=drift["force_x"] + rng.uniform(-0.02, 0.02),
                    force_y=drift["force_y"] + rng.uniform(-0.02, 0.02),
                    force_z=drift["force_z"] + rng.uniform(-0.02, 0.02),
                    torque_x=drift["torque_x"] + rng.uniform(-0.02, 0.02),
                    torque_y=drift["torque_y"] + rng.uniform(-0.02, 0.02),
                    torque_z=drift["torque_z"] + rng.uniform(-0.02, 0.02),
                    pos_x=drift["pos_x"] + rng.uniform(-0.02, 0.02),
                    pos_y=drift["pos_y"] + rng.uniform(-0.02, 0.02),
                    pos_z=drift["pos_z"] + rng.uniform(-0.02, 0.02),
                    temp_c=drift["temp_c"] + rng.uniform(-0.02, 0.02),
                    smoke_ppm=drift["smoke_ppm"] + rng.uniform(-0.02, 0.02),
                )
            )

        records.append(
            WindowedFusedRecord(
                timestamp_ns=ts,
                rgb_path="",
                depth_path=None,
                window_size_s=0.1,
                n_samples=len(sensor_window),
                sensor_window=sensor_window,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Metrics polling
# ---------------------------------------------------------------------------


def _extract_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract key counters from a runtime-metrics response."""
    return {
        "anomalies": raw.get("anomalies_detected", raw.get("anomaly_count", 0)),
        "plans_executed": raw.get("plans_executed", raw.get("completed_plans", 0)),
        "emergency_stop": raw.get("emergency_stop", False),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a CortexGuard anomaly scenario to a running edge service.",
    )
    parser.add_argument(
        "--list", action="store_true", help="Print all available scenarios and exit"
    )
    parser.add_argument("--scenario", default="S0.1", help="Scenario ID to stream (default: S0.1)")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8080/api/v1/ingest",
        help="Ingest endpoint URL (default: http://localhost:8080/api/v1/ingest)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Records per second (default: 1.0)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the stream (0 = infinite, default: 1)",
    )
    parser.add_argument(
        "--scenarios-file",
        default="data/anomaly_scenarios.yaml",
        help="Path to anomaly_scenarios.yaml (default: data/anomaly_scenarios.yaml)",
    )
    return parser.parse_args()


def _print_scenarios(scenarios: dict[str, Scenario]) -> None:
    print(f"{'ID':<8}  {'Tier':<5}  {'Title':<40}  Expected outcome")
    print("-" * 90)
    for sid, s in sorted(scenarios.items()):
        print(f"{sid:<8}  {s.tier:<5}  {s.title:<40}  {s.expected_outcome}")


def _base_url(endpoint: str) -> str:
    """Derive base URL from ingest endpoint (strip path)."""
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    return f"{parsed.scheme}://{parsed.netloc}"


def _check_reachable(endpoint: str) -> bool:
    """Return True if the endpoint host is reachable (best-effort HEAD/GET)."""
    base = _base_url(endpoint)
    result = _get_json(f"{base}/healthz")
    if result is not None:
        return True
    # healthz may not return JSON — try a raw request
    try:
        import urllib.request

        urllib.request.urlopen(f"{base}/healthz", timeout=5).close()
        return True
    except Exception:
        return False


def main() -> int:
    """Entry point.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    args = _parse_args()

    # Resolve scenarios file relative to repo root when run from any CWD
    scenarios_path = Path(args.scenarios_file)
    if not scenarios_path.exists():
        # Try relative to the script's parent directory (repo root)
        alt = Path(__file__).parent.parent / args.scenarios_file
        if alt.exists():
            scenarios_path = alt

    try:
        scenarios = load_scenarios(scenarios_path)
    except FileNotFoundError:
        print(f"ERROR: scenarios file not found: {scenarios_path}", file=sys.stderr)
        return 1

    if args.list:
        _print_scenarios(scenarios)
        return 0

    scenario_id: str = args.scenario
    if scenario_id not in scenarios:
        print(
            f"ERROR: scenario '{scenario_id}' not found. "
            f"Available: {', '.join(sorted(scenarios))}",
            file=sys.stderr,
        )
        return 1

    scenario = scenarios[scenario_id]
    repeat = args.repeat  # 0 = infinite
    print(f"Scenario : {scenario.scenario_id} — {scenario.title}")
    print(f"Tier     : {scenario.tier}")
    print(f"Outcome  : {scenario.expected_outcome}")
    print(f"Endpoint : {args.endpoint}")
    print(f"Rate     : {args.rate} records/s")
    print(f"Repeat   : {'infinite' if repeat == 0 else repeat}")
    print()

    # Connectivity check
    if not _check_reachable(args.endpoint):
        print(f"ERROR: edge service unreachable at {_base_url(args.endpoint)}", file=sys.stderr)
        return 1

    # Build stream
    baseline = _make_synthetic_baseline(n_records=20, window_size=5, seed=scenario.seed or 42)
    engine = ChaosEngine(scenario)
    stream = list(engine.inject(iter(baseline)))

    total = len(stream)
    sleep_s = 1.0 / max(args.rate, 0.01)
    base = _base_url(args.endpoint)
    metrics_url = f"{base}/runtime-metrics"

    cycle = 0
    final_metrics: dict[str, Any] = {}

    while repeat == 0 or cycle < repeat:
        cycle += 1
        last_status_t = time.monotonic()

        for idx, record in enumerate(stream, start=1):
            payload = record.model_dump()
            try:
                status = _post_json(args.endpoint, payload)
            except Exception as exc:
                print(f"ERROR: POST failed on record {idx}: {exc}", file=sys.stderr)
                return 1

            if status >= 400:
                print(f"WARNING: record {idx} returned HTTP {status}", file=sys.stderr)

            # Poll metrics roughly every second
            now = time.monotonic()
            if now - last_status_t >= 1.0 or idx == total:
                raw = _get_json(metrics_url) or {}
                m = _extract_metrics(raw)
                final_metrics = m
                cycle_label = f"cycle {cycle} " if repeat != 1 else ""
                print(
                    f"[{cycle_label}record {idx:>2}/{total}]  "
                    f"anomalies={m['anomalies']}  "
                    f"plans_executed={m['plans_executed']}  "
                    f"emergency_stop={m['emergency_stop']}"
                )
                last_status_t = now

            if idx < total:
                time.sleep(sleep_s)

    print()
    print("=== Demo complete ===")
    print(f"Records streamed : {total * cycle}")
    print(f"Anomalies        : {final_metrics.get('anomalies', 'n/a')}")
    print(f"Plans executed   : {final_metrics.get('plans_executed', 'n/a')}")
    print(f"Emergency stop   : {final_metrics.get('emergency_stop', 'n/a')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
