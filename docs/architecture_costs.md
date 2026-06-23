# Architecture Costs

---

## Edge Hardware

The edge tier is designed to run on an NVIDIA Jetson Orin (target deployment platform).

| Component | Spec | Estimated Cost |
|---|---|---|
| Jetson Orin NX 16GB | 16 TOPS, 10W–25W TDP | ~$500 |
| Jetson AGX Orin 64GB | 275 TOPS, up to 60W TDP | ~$999 |

The AGX Orin is recommended for running the full stack including the LLM policy engine (7B model) locally. The NX is sufficient in mock/rules-only policy mode.

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

The LLM policy engine runs `Qwen/Qwen2.5-7B-Instruct` locally on the edge device.

| Mode | Hardware | Inference Time (est.) |
|---|---|---|
| GPU (Jetson AGX Orin) | ~60W | ~200–500 ms per policy call |
| CPU fallback | high power draw | several seconds: not suitable for production |

**Cost**: Once the model is downloaded, inference is free (on-device). No per-call API cost.

Policy calls are only triggered for anomalies that pass the Z-score threshold (`anomaly_threshold = 5.0`) and are not handled by rules-based dispatch, so call frequency is low in normal operation.

---

## Cloud Deliberative Planner (Implemented)

The cloud tier is implemented as a FastAPI service running a LangGraph planning workflow with Qdrant vector search and a pluggable LLM backend.

### Deliberative Inference (Cloud LLM)

| Backend | Model | Cost |
|---|---|---|
| Groq (default) | `llama-3.3-70b-versatile` | Free tier available; paid tier ~$0.0006/1K tokens |
| Anthropic | `claude-haiku-4-5-20251001` | ~$0.00025/1K input tokens |
| OpenRouter | configurable | varies by model |

Escalations are rare by design, the edge handles the majority of cases locally. Expected volume: < 10 escalations/device/day in normal operation.

### Infrastructure

| Component | Service | Notes |
|---|---|---|
| Incident store | SQLite (local) / replaceable | Persistent history of all escalations |
| Vector store | Qdrant (self-hosted) | 384-dim MiniLM embeddings |
| Embedder | `all-MiniLM-L6-v2` (CPU) | ~50ms per embed on CPU |

### Model Lifecycle (AWS SageMaker)

| Resource | Instance | Estimated Cost |
|---|---|---|
| Step classifier endpoint | `ml.t2.medium` (CPU) | ~$0.05/hr |
| Retraining pipeline | `ml.m5.large` (CPU) | ~$0.10/hr per run |
| Model registry (artifacts) | S3 per GB/month | ~$0.023/GB |

The retraining pipeline runs weekly on a schedule. The `StepClassifierClient` on the edge calls the SageMaker endpoint via the cloud API (`POST /api/v1/classify`). Data drift monitoring and champion/challenger Lambda functions automate model rollback and promotion.

---

## Trade-off Summary

| Mode | Latency | Cost | Capability |
|---|---|---|---|
| Edge only (rules-based) | < 100 ms | Hardware only | Handles known anomaly patterns |
| Edge + local LLM (Qwen2.5-7B) | 100–500 ms | Hardware only | Handles novel anomalies locally |
| Edge + cloud escalation (MaydayAgent) | 1–5 s | LLM API cost per escalation | Handles complex / unknown failures |
| Fleet-level coordination (future) | 1–10 s | Cloud compute + LLM API | Cross-device reasoning, retraining, XAI |

**Design principle**: never block safety-critical decisions on cloud. Edge acts immediately; cloud reconciles and improves policies asynchronously.
