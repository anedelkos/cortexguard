# CortexGuard Edge — Metrics Specification

## Overview

CortexGuard Edge exposes a **minimal, stable, low‑cardinality** set of Prometheus metrics.
These metrics provide high‑value operational visibility across the entire edge pipeline while avoiding cardinality explosions and unnecessary noise.

All metrics are defined in:

```text
cortexguard.edge.utils.metrics
```

This document defines:

- metric **names**
- metric **types**
- metric **labels** (and allowed values)
- metric **purpose**
- metric **cardinality constraints**
- metric **emission points** in the system

---

## 1. `cortexguard_component_duration_ms`

**Type:** Histogram

**Labels:**

- `component` (single, fixed, low‑cardinality label)

**Allowed label values:**

| Component value              | Meaning                                           |
|-----------------------------|---------------------------------------------------|
| `fusion_process_record`     | Fusion pipeline latency per record                |
| `estimator_update`          | OnlineLearnerStateEstimator update latency        |
| `anomaly_detector_tick`     | Full anomaly detection tick latency               |
| `policy_generation`         | PolicyAgent policy generation latency             |
| `policy_publish`            | Policy publication overhead                       |
| `policy_agent_tick`         | PolicyAgent tick latency                          |
| `mayday`                    | Full Mayday escalation latency                    |
| `mayday_attempt`            | Individual cloud escalation attempt latency       |
| `orchestrator_tick`         | Orchestrator control loop tick latency            |

**Purpose:**
Measure latency of major CortexGuard subsystems in milliseconds.

**Cardinality constraints:**

- `component` **must** be one of the fixed values above.
- **No dynamic values** allowed.
- **No per‑plan**, **per‑anomaly**, or **per‑detector** labels.

**Emitted by:**

- Fusion Engine (`process_record`)
- Estimator (`update`)
- AnomalyDetector (`_run_tick`)
- PolicyAgent (`_generate_policy_for_anomaly`, `_publish_policy`, `_process_active_anomalies_tick`)
- MaydayAgent (`send_escalation` and helpers)
- Orchestrator (`_run_loop`)

---

## 2. `cortexguard_anomalies_total`

**Type:** Counter

**Labels:**
None

**Purpose:**
Count anomalies emitted by the AnomalyDetector.

**Cardinality constraints:**

- No labels.
- No per‑anomaly keys.

**Emitted by:**

- `AnomalyDetector._emit_and_store_active_anomalies` (for newly active anomalies)

---

## 3. `cortexguard_detector_failures_total`

**Type:** Counter

**Labels:**
None

**Purpose:**
Count sub‑detector exceptions during anomaly detection.

**Cardinality constraints:**

- No labels.
- No per‑detector names.

**Emitted by:**

- `AnomalyDetector._run_detectors` (when a detector raises an exception)

---

## 4. `cortexguard_policy_escalations_total`

**Type:** Counter

**Labels:**
None

**Purpose:**
Count remediation policies that required escalation to Mayday.

**Cardinality constraints:**

- No labels.
- No per‑policy IDs.

**Emitted by:**

- `PolicyAgent._escalate_to_mayday_agent` (when escalation is triggered)
- Optionally mirrored in `MaydayAgent.send_escalation` if you choose to count attempts there as well.

---

## 5. `cortexguard_active_anomalies`

**Type:** Gauge

**Labels:**
None

**Purpose:**
Track the number of currently active anomalies.

**Cardinality constraints:**

- No labels.
- Gauge value is a simple integer.

**Emitted by:**

- AnomalyDetector, when updating its active anomaly key set and syncing with the blackboard.

---

## 6. `cortexguard_plan_queue_length`

**Type:** Gauge

**Labels:**
None

**Purpose:**
Track the number of pending plans in the Orchestrator queue.

**Cardinality constraints:**

- No labels.

**Emitted by:**

- Orchestrator, whenever plans are enqueued or dequeued (queue length changes).

---

## 7. `cortexguard_estimator_confidence`

**Type:** Gauge

**Labels:**
None

**Purpose:**
Expose the latest confidence value from the `OnlineLearnerStateEstimator`.

**Cardinality constraints:**

- No labels.
- Value is constrained to the range \([0.0, 1.0]\).

**Emitted by:**

- `Estimator.update`, immediately after computing confidence.

---

## 8. `cortexguard_http_requests_total`

**Type:** Counter

**Labels:**

- `method` (HTTP method)
- `status_code` (HTTP response status code as string)

**Allowed label values:**

| Label | Allowed values |
|-------|---------------|
| `method` | `POST` |
| `status_code` | `202`, `422`, `500` |

**Purpose:**
Count HTTP ingestion requests by method and outcome status code.

**Cardinality constraints:**

- `method` is always `POST` for the ingest endpoint.
- `status_code` is one of a fixed set of HTTP status strings — never dynamic.

**Emitted by:**

- `api/ingestion.py` ingest route handler (202, 500)
- `runtime.py` RequestValidationError exception handler (422)

---

## 9. `cortexguard_http_request_duration_ms`

**Type:** Histogram

**Labels:**
None

**Purpose:**
Measure HTTP ingestion endpoint latency in milliseconds.

**Cardinality constraints:**

- No labels.
- Buckets: 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000 ms.

**Emitted by:**

- `api/ingestion.py` ingest route handler (per-request, in finally block)

---

## 10. `cortexguard_llm_circuit_open`

**Type:** Gauge

**Labels:**
None

**Purpose:**
Expose the LLM circuit breaker state. Value is `1` when the circuit is open (tripped after consecutive failures) and `0` when closed (operating normally).

**Cardinality constraints:**

- No labels.
- Value is always 0 or 1.

**Emitted by:**

- `PolicyAgent._record_llm_success` (sets to 0)
- `PolicyAgent._record_llm_failure` (sets to 1 when threshold reached)

---

## 11. `cortexguard_llm_requests_total`

**Type:** Counter

**Labels:**

- `outcome` (result of the LLM call)

**Allowed label values:**

| Outcome value | Meaning |
|--------------|---------|
| `success` | LLM returned a valid policy |
| `failure` | LLM raised a non-timeout exception |
| `timeout` | LLM call exceeded the configured timeout |
| `circuit_skipped` | LLM call was skipped because the circuit breaker is open |

**Purpose:**
Count LLM policy generation requests by outcome to enable RED method coverage for the LLM subsystem.

**Cardinality constraints:**

- `outcome` **must** be one of the four fixed values above.
- No per-anomaly or per-request identifiers.

**Emitted by:**

- `PolicyAgent._generate_unknown_medium_policy` (success, failure, timeout, circuit_skipped)

---

## 12. `cortexguard_mayday_escalations_total`

**Type:** Counter

**Labels:**

- `outcome` (result of the escalation attempt)

**Allowed label values:**

| Outcome value | Meaning |
|--------------|---------|
| `success` | Cloud returned a plan |
| `timeout` | Cloud call timed out |
| `error` | Cloud call raised a non-timeout exception |
| `exhausted` | All retry attempts exhausted without success |

**Purpose:**
Count cloud escalation attempts by outcome to expose MaydayAgent's internal counters as a Prometheus metric.

**Cardinality constraints:**

- `outcome` **must** be one of the four fixed values above.
- No per-escalation or per-device identifiers.

**Emitted by:**

- `MaydayAgent._attempt_escalation` (success, timeout, error)
- `MaydayAgent.send_escalation` (exhausted, after retry loop)

---

## 13. `cortexguard_mayday_consecutive_failures`

**Type:** Gauge

**Labels:**
None

**Purpose:**
Track the current number of consecutive cloud escalation failures. Resets to 0 on success.

**Cardinality constraints:**

- No labels.
- Value is a non-negative integer.

**Emitted by:**

- `MaydayAgent._attempt_escalation` (set to 0 on success, set to `_consecutive_failures` on timeout/error)

---

## 14. `cortexguard_plans_total`

**Type:** Counter

**Labels:**

- `status` (terminal status of the plan)

**Allowed label values:**

| Status value | Meaning |
|-------------|---------|
| `completed` | Plan executed all steps successfully |
| `failed` | Plan reached a terminal failure state |

**Purpose:**
Count plans by their terminal outcome to enable plan success rate SLOs.

**Cardinality constraints:**

- `status` **must** be one of the two fixed values above.
- No per-plan or per-anomaly identifiers.

**Emitted by:**

- `Orchestrator._advance_plan_or_handle_failure` (at both terminal state transitions)

---

## 15. `cortexguard_steps_total`

**Type:** Counter

**Labels:**

- `outcome` (terminal outcome of the step)

**Allowed label values:**

| Outcome value | Meaning |
|--------------|---------|
| `completed` | Step executed and completed successfully |
| `retry_exhausted` | Step permanently failed after all retry attempts |
| `aborted` | Step was aborted due to emergency stop or safety flag |

**Purpose:**
Count step executions by outcome to observe step-level failure rates and retry exhaustion.

**Cardinality constraints:**

- `outcome` **must** be one of the three fixed values above.
- No per-step or per-plan identifiers.

**Emitted by:**

- `StepExecutor.execute_step` (at each of the three terminal branches)

---

## Design principles

**Low cardinality**

All metrics avoid:

- dynamic labels
- per‑anomaly labels
- per‑plan labels
- per‑detector labels
- per‑feature labels

**High value**

Each metric answers a clear operational question:

- *Is the system slow or overloaded?*
  - `cortexguard_component_duration_ms`

- *Are detectors failing?*
  - `cortexguard_detector_failures_total`

- *Are anomalies spiking or persisting?*
  - `cortexguard_anomalies_total`, `cortexguard_active_anomalies`

- *Is the estimator still confident?*
  - `cortexguard_estimator_confidence`

- *Is the orchestrator backlogged?*
  - `cortexguard_plan_queue_length`

- *Are we escalating too often?*
  - `cortexguard_policy_escalations_total`

**Stable contract**

This spec defines the **only** metrics exposed by CortexGuard Edge unless a future change explicitly updates this document. Any new metric must:

- justify its purpose,
- respect cardinality constraints,
- and be added to this spec before being merged.
