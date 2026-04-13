# Architecture Costs

---

## Edge Hardware

The edge tier is designed to run on an NVIDIA Jetson Orin (target deployment platform).

| Component | Spec | Estimated Cost |
|---|---|---|
| Jetson Orin NX 16GB | 16 TOPS, 10W–25W TDP | ~$500 |
| Jetson AGX Orin 64GB | 275 TOPS, up to 60W TDP | ~$999 |

The AGX Orin is recommended for running the full stack including `MistralPolicyEngine` (7B model) locally. The NX is sufficient in mock/rules-only policy mode.

---

## Latency Budget (Edge)

Based on runtime configuration in `RuntimeConfig`:

| Stage | Interval / Latency | Notes |
|---|---|---|
| Sensor ingestion (fusion) | 100 ms (10 Hz) | `sensor_fusion_rate = 0.1` |
| Anomaly detection tick | 1,000 ms | `anomaly_check_interval = 1.0s` |
| Orchestrator tick | 100 ms | `orchestrator_tick_interval = 0.1s` |
| Step executor poll | 50 ms | `executor_poll_interval = 0.05s` |
| Step retry delay | 500 ms | `executor_retry_delay = 0.5s`, max 3 retries |
| SafetyAgent E-STOP | < 100 ms | Evaluated every orchestrator tick |
| Cloud escalation timeout | 5,000 ms | `MaydayAgent.timeout_seconds = 5.0` |

**Hard constraint**: Safety-critical actions (E-STOP, PAUSE) must complete within one orchestrator tick (100 ms). Cloud is never in the loop for these.

---

## LLM Policy Engine

The `MistralPolicyEngine` runs `mistralai/Mistral-7B-Instruct-v0.2` locally on the edge device.

| Mode | Hardware | Inference Time (est.) |
|---|---|---|
| GPU (Jetson AGX Orin) | ~60W | ~200–500 ms per policy call |
| CPU fallback | high power draw | several seconds — not suitable for production |

**Cost**: Once the model is downloaded, inference is free (on-device). No per-call API cost.

Policy calls are only triggered for anomalies that pass the Z-score threshold (`anomaly_threshold = 5.0`) and are not handled by rules-based dispatch, so call frequency is low in normal operation.

---

## AWS Cloud (Planned)

> The deliberative cloud layer is not yet implemented. Estimates below are for planning purposes.

### Training Pipeline (AWS SageMaker)

| Resource | Instance | Estimated Cost |
|---|---|---|
| Detector fine-tuning | `ml.g4dn.xlarge` (T4 GPU) | ~$0.74/hr |
| Meta-model retraining | `ml.m5.large` (CPU) | ~$0.10/hr |
| Storage (S3 episodes) | per GB/month | ~$0.023/GB |

Training runs are expected to be periodic (nightly or on-demand), not continuous.

### Deliberative Inference (Cloud LLM)

| Scenario | Model | Estimated Cost |
|---|---|---|
| Recovery planning (escalated anomalies) | Claude Haiku / GPT-4o-mini | ~$0.001–$0.01 per escalation |
| Explanation agent (XAI) | Claude Haiku | ~$0.001 per query |

Escalations are rare by design — the edge handles the majority of cases locally. Expected volume: < 10 escalations/device/day in normal operation.

---

## Trade-off Summary

| Mode | Latency | Cost | Capability |
|---|---|---|---|
| Edge only (rules-based) | < 100 ms | Hardware only | Handles known anomaly patterns |
| Edge + local LLM (Mistral) | 100–500 ms | Hardware only | Handles novel anomalies locally |
| Edge + cloud escalation (MaydayAgent) | 1–5 s | AWS inference cost | Handles complex / unknown failures |
| Full deliberative cloud layer (planned) | 1–10 s | AWS compute + LLM API | Fleet-level reasoning, retraining, XAI |

**Design principle**: never block safety-critical decisions on cloud. Edge acts immediately; cloud reconciles and improves policies asynchronously.
