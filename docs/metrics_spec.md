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
