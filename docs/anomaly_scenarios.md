# 🍳 KitchenWatch Scenario Catalogue (Edge Anomaly Simulation)

Each scenario has:
**ID**, **Title**, **Anomalies Injected**, **Goal**, **Expected Edge Response**, **Outcome Metric**

---

## 🔹 Tier 0 — Safety-Critical (Immediate Stop / Hard Fault)

### S0.1 — Human in Safety Radius
- **Anomalies:** `human_intrusion(severity=critical, distance_m=0.2)`
- **Goal:** Verify fast detection of human presence and immediate stop.
- **Expected Response:** Robot halts all motion within 200 ms, logs safety event, sends status to cloud.
- **Metric:** detection latency, false negatives = 0.

### S0.2 — Overheat + Smoke Combo
- **Anomalies:** `temp_spike(+70°C)`, `visual_smoke(opacity=0.8)`
- **Goal:** Test multi-sensor correlation for fire-like conditions.
- **Expected Response:** Hard stop, trigger cooling/fan sequence, raise Tier0 alert.
- **Metric:** multi-sensor fusion correctness, escalation latency.

### S0.3 — Impact During Motion
- **Anomalies:** `ft_spike(20× amplitude)`
- **Goal:** Ensure force-based reflex trigger cuts actuator power.
- **Expected Response:** Emergency stop + log force vector.
- **Metric:** stop latency < 100 ms.

---

## 🔹 Tier 1 — Recoverable but Critical (Needs Cloud or Supervisor Input)

### S1.1 — Repeated Misgrasp (Re-Grip Failure)
- **Anomalies:** `slip(force_pct=40%, drift_mm=35)` repeated 3×.
- **Goal:** Test retry logic and escalation after multiple failures.
- **Expected Response:** Local retries exhausted → send remediation request to cloud.
- **Metric:** retry count vs escalation timing.

### S1.2 — Camera Occlusion (Persistent)
- **Anomalies:** `vision_occlusion(area_pct=70%, duration_s=5)`
- **Goal:** Detect long-term occlusion → degrade gracefully.
- **Expected Response:** Vision flag “unavailable”; fusion continues with F/T; trigger “vision degraded” warning.
- **Metric:** detection accuracy, correct fallback to proprioception.

### S1.3 — Human Crosses Robot Path
- **Anomalies:** `human_path_conflict(TTI_s=0.4)`
- **Goal:** Predict collision, preempt motion safely.
- **Expected Response:** Slowdown or replan; confirm predictive detection.
- **Metric:** preemption correctness, TTI margin ≥ 0.3 s.

---

## 🔹 Tier 2 — Operational Anomalies (Recoverable Locally)

### S2.1 — Burger Slip on Grill
- **Anomalies:** `ext_motion(delta_mm=60)`, optional `vision_occlusion(area_pct=30%)`
- **Goal:** Verify object drift detection and auto “re-grip_patty” trigger.
- **Expected Response:** Local step classifier fails current step; re-executes re-grip plan.
- **Metric:** recovery success rate.

### S2.2 — Force/Torque Micro Spikes
- **Anomalies:** `ft_spike(3× amplitude, duration_ms=50)`
- **Goal:** Test false-positive filtering (should *not* halt).
- **Expected Response:** Tolerate blip, continue.
- **Metric:** false positive count per hour.

### S2.3 — Temporary Sensor Freeze
- **Anomalies:** `sensor_freeze(duration_s=1)`
- **Goal:** Validate temporal redundancy and timeout handling.
- **Expected Response:** Hold last good state, retry; no cascade failure.
- **Metric:** missed-step percentage.

---

## 🔹 Tier 3 — Environmental & Communication Faults

### S3.1 — Network Jitter
- **Anomalies:** `comm_lag(latency_ms=500, packet_loss_pct=5)`
- **Goal:** Confirm edge autonomy under degraded cloud link.
- **Expected Response:** Edge maintains operation, logs warning.
- **Metric:** continuity under loss, task completion %.

### S3.2 — Prolonged Latency (Cloud Unreachable)
- **Anomalies:** `comm_lag(latency_ms>1000, packet_loss_pct=20)`
- **Goal:** Ensure safe degraded mode when cloud comms fail.
- **Expected Response:** Switch to local-only mode; store events for upload.
- **Metric:** correct fallback path activation.

### S3.3 — Human Loitering Near Workspace
- **Anomalies:** `human_intrusion(distance_m=1.0, duration_s=10)`
- **Goal:** Verify “cautious mode” behavior, slower motion, continuous recheck.
- **Expected Response:** speed reduction (≤ 50%), maintain awareness.
- **Metric:** latency of speed modulation.

---

## 🔹 Tier 4 — Compound Scenarios (Stress / End-to-End)

### S4.1 — Multi-Fault: Slip + Occlusion + Comm Lag
- **Anomalies:** `slip`, `vision_occlusion(40%)`, `comm_lag(500ms)`
- **Goal:** Evaluate fusion resilience with overlapping faults.
- **Expected Response:** Maintain stable state estimation; escalate only if unresolvable.
- **Metric:** fusion accuracy under compound anomaly.

### S4.2 — Fire + Human Intrusion
- **Anomalies:** `temp_spike(+60°C)`, `visual_smoke(0.6)`, `human_intrusion(0.3m)`
- **Goal:** Validate that human safety overrides all; plan preemption ordering.
- **Expected Response:** Stop motion → alert → safety broadcast.
- **Metric:** correct safety prioritization (human > environment).

### S4.3 — Grill Overheat While Re-Stacking
- **Anomalies:** `temp_spike(+40°C)` during action sequence `re-stack_ingredients`
- **Goal:** Verify the step classifier halts current plan and prioritizes cooling task.
- **Expected Response:** Recipe paused, fire-mitigation plan triggered.
- **Metric:** plan interruption correctness.

---

## 🔹 Tier 5 — Long-Term Drift / Subtle Anomalies (for ML model robustness)

### S5.1 — Gradual F/T Bias Drift
- **Anomalies:** add bias +0.5 N per min over 10 min.
- **Goal:** Evaluate trend-based detectors.
- **Expected Response:** Model should detect calibration drift, not confuse with impact.
- **Metric:** time-to-detection, false escalation rate.

### S5.2 — Lighting Change in Scene
- **Anomalies:** gradual brightness drop 50% over 30 s.
- **Goal:** Test vision normalization.
- **Expected Response:** Maintain recognition confidence.
- **Metric:** drop in classification confidence ≤ 10%.

### S5.3 — Temperature Sensor Noise Burst
- **Anomalies:** random spikes ±10°C for 2 s.
- **Goal:** Validate temporal smoothing in the anomaly model.
- **Expected Response:** Smooth values; no false alarm.
- **Metric:** spurious alarm count.

---

## 📈 Metadata Conventions for Each Scenario

| Field | Example | Purpose |
|--------|----------|----------|
| `scenario_id` | "S2.1" | unique reference |
| `seed` | 42 | deterministic replay |
| `duration_s` | 30 | total simulation time |
| `baseline_recipe` | "make_burger" | context for actions |
| `anomalies` | list[dict] | injected anomalies |
| `expected_outcome` | "recovery_success" / "hard_stop" / "escalate_to_cloud" | expected result |
| `tier` | 0–5 | safety priority |
