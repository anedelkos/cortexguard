# Testing Strategy

---

## Overview

CortexGuard uses a three-tier test pyramid: unit → integration → e2e. Tests are separated by speed and dependency requirements, allowing fast feedback during development while still covering realistic end-to-end behaviour.

---

## Test Tiers

### Unit Tests (`tests/unit/`)

Fast, isolated, no external dependencies. Each test runs against a single component with all dependencies mocked or stubbed.

**Coverage target**: 80% minimum (enforced by `pytest-cov`, `fail_under = 80`).

**Run with**: `task test-unit`

Key areas covered:

| Area | Test File(s) |
|---|---|
| Anomaly detector ensemble | `anomaly_detectors/test_*.py` |
| Edge fusion (EMA, sensor smoothing) | `test_edge_fusion.py` |
| Orchestrator (plan scheduling, preemption) | `test_orchestrator.py` |
| Step executor (execution loop, retries) | `test_step_executor.py` |
| Safety agent (rules, E-STOP logic) | _(via orchestrator integration)_ |
| Policy agent (rules-based + LLM dispatch) | `test_policy_agent.py` |
| MaydayAgent (escalation, retry/backoff) | `test_mayday_agent.py` |
| Arbiter (capability validation, motion gating) | `test_arbiter.py` |
| Online learner / state estimator | `test_online_learner_state_estimator.py`, `test_river_online_learner.py` |
| Mistral policy engine (mocked LLM) | `test_mistral_policy_engine.py` |
| Runtime wiring | `test_runtime.py` |
| Simulation models and streamers | `simulation/test_*.py` |

---

### Integration Tests (`tests/integration/`)

Moderate speed (< 30s). Test subsystems working together with real async loops and injected fused data. Cloud LLM calls are mocked. Require `@pytest.mark.integration` or `@pytest.mark.asyncio`.

**Run with**: `task test-integration` (excludes `llm_slow`)

| Test File | What It Tests |
|---|---|
| `test_anomaly_scenarios.py` | Full edge runtime + `ChaosEngine` anomaly injection; validates detection and safety response end-to-end |
| `test_orchestrator_safety_integration.py` | Orchestrator + SafetyAgent interaction under anomalous conditions |
| `test_plan_execution.py` | Plan submission, preemption, and execution via Orchestrator + StepExecutor |
| `test_mayday_agent.py` | MaydayAgent escalation with mock cloud client |
| `test_modalities_fuser.py` | Simulation fusion pipeline with real sensor data |
| `test_manifest_lookup.py` | Manifest loading and trial resolution |
| `test_mistral_policy_engine_real_llm.py` | Real LLM call — marked `@pytest.mark.llm_slow`, excluded from default runs |

#### Chaos Engine

The `ChaosEngine` (`tests/integration/chaos_engine.py`) injects anomalies into a baseline `WindowedFusedRecord` stream according to a `Scenario` loaded from `data/anomaly_scenarios.yaml`. Scenarios are classified by `tier` (0–5) and define:

- **Anomaly type**: `human_intrusion`, `temp_spike`, `visual_smoke`, `slip`, `sensor_freeze`, `vision_occlusion`, `comm_lag`
- **Expected outcome**: what the system should do (E-STOP, PAUSE, escalate, etc.)

This is the primary mechanism for testing the full detection → fusion → safety pipeline with realistic, reproducible fault injection.

---

### E2E Tests (`tests/e2e/`)

Slow, realistic. Spin up the full runtime or simulate a stream end-to-end.

**Run with**: `task test-e2e`

| Test File | What It Tests |
|---|---|
| `test_pipeline_run.py` | LocalStreamer → LocalReceiver pipeline with a real manifest and fused JSONL |
| `test_simulate_stream.py` | Simulate stream ingestion against a live edge receiver |

---

## Markers

| Marker | Meaning | Included in `task test` |
|---|---|---|
| _(none)_ | Standard unit/integration test | Yes |
| `@pytest.mark.asyncio` | Async test — requires `pytest-asyncio` | Yes |
| `@pytest.mark.integration` | Requires external resources | Yes (unless also `llm_slow`) |
| `@pytest.mark.llm_slow` | Real LLM / hardware call, slow | No — use `task test-llm-slow` |

---

## Coverage Configuration

Configured in `pyproject.toml`:

- **Branch coverage** enabled
- **Minimum**: 80% (`fail_under = 80`)
- **Excluded from coverage**: `models/`, `core/mocks/`, `core/interfaces/`, `logging_config.py` — these are data containers, mock stubs, or protocols unlikely to benefit from line coverage

---

## Task Reference

| Task | What Runs |
|---|---|
| `task test` | Unit + integration, excludes `llm_slow`, with coverage |
| `task test-unit` | Unit tests only |
| `task test-integration` | Integration tests only, excludes `llm_slow` |
| `task test-llm-slow` | Real LLM tests only — for CI/nightly |
| `task test-e2e` | E2E tests only |

Set `PYTHONPATH=src` when running `pytest` directly (tasks do this automatically).

---

## Running a Single Test

```bash
# Single file
PYTHONPATH=src pytest tests/unit/edge/test_orchestrator.py -v

# Single test
PYTHONPATH=src pytest tests/unit/edge/test_orchestrator.py::test_name -v

# Chaos engine scenarios
PYTHONPATH=src pytest tests/integration/test_anomaly_scenarios.py -v
```

---

## What Is Not Tested

- **Real hardware** — actuator commands use `MockController`; no hardware-in-the-loop tests
- **Real vision inference** — vision detectors use injected `SceneGraph` data; no camera/model integration tests
- **LLM policy in CI** — `MistralPolicyEngine` runs in mock mode by default; real LLM tests are gated behind `llm_slow`
