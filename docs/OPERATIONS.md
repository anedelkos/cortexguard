# CortexGuard Edge — Operations Reference

Audience: deploying, configuring, or debugging the edge service.

---

## Environment Variables

All variables are optional with the defaults shown. Set them in your shell, `.env` file, or Docker Compose `environment:` block.

| Variable | Default | Purpose                                                                                                              |
|----------|---------|----------------------------------------------------------------------------------------------------------------------|
| `DEVICE_ID` | `mock_01` | Device identity tag in logs and traces                                                                               |
| `RUNTIME_PROFILE` | `default` | Runtime profile selector                                                                                             |
| `POLICY_MODEL_ID` | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model ID for LLM policy engine                                                                           |
| `POLICY_USE_MOCK` | `true` | `false` to enable real Mistral-7B inference (requires full deps + GPU recommended)                                   |
| `POLICY_REMEDIATION_COOLDOWN_S` | `30.0` | Minimum seconds between remediation policy generations for the same anomaly                                          |
| `LLM_TIMEOUT_S` | `30.0` | Per-call timeout in seconds for the LLM policy engine                                                                |
| `LLM_FAILURE_THRESHOLD` | `3` | Consecutive LLM failures before the circuit breaker opens                                                            |
| `LLM_COOLDOWN_S` | `60.0` | Duration in seconds the LLM circuit breaker stays open before resetting                                              |
| `PERSISTENCE_ENABLED` | `false` | Enable periodic blackboard snapshots to disk                                                                         |
| `PERSISTENCE_FILE_PATH` | `/var/lib/cortexguard/blackboard.json` | Blackboard snapshot location                                                                                         |
| `PERSISTENCE_SNAPSHOT_INTERVAL` | `5.0` | Seconds between snapshots                                                                                            |
| `OTLP_ENDPOINT` | `http://tempo:4318/v1/traces` | OpenTelemetry trace collector endpoint                                                                               |
| `LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)                                                                      |
| `LOG_JSON` | `true` | JSON structured logs in prod; set `false` for readable local output                                                  |
| `INGEST_RATE_LIMIT` | `100/second` | Rate limit for `POST /api/v1/ingest` per client IP (slowapi format, e.g. `200/second`, `1000/minute`)                |
| `FUSION_FORCE_MIN_N` | `0.0` | Force sensor floor in Newtons (values below are clamped)                                                             |
| `FUSION_FORCE_DROP_PCT` | `100.1` | Force drop detection threshold as % change                                                                           |
| `FUSION_DRIFT_FAIL_MM` | `10.0` | Position drift failure threshold in mm                                                                               |
| `FUSION_SMOKE_PPM_THRESHOLD` | `50.0` | Smoke sensor threshold in PPM above which smoke is flagged                                                           |
| `FUSION_EXPECTED_PERIOD_MS` | `50` | Expected sensor window arrival interval in milliseconds                                                              |
| `FUSION_SOFT_DEGRADE_MS` | `200` | Arrival lag threshold in milliseconds above which timing is marked degraded                                          |
| `FUSION_MAX_GAP_MS` | `500` | Maximum tolerated arrival gap in milliseconds before data is considered stale                                        |
| `SAFETY_RADIUS_M` | `0.5` | Minimum safe distance in metres between hardware and detected humans                                                 |
| `DETECTOR_TEMP_THRESHOLD_C` | `70.0` | Temperature threshold in °C above which an overheat anomaly is raised                                                |
| `DETECTOR_Z_SCORE_THRESHOLD` | `5.0` | Z-score threshold above which the statistical impulse detector fires                                                 |
| `ESTIMATOR_SIGMA_THRESHOLD` | `3.0` | Standard deviation threshold used by the online state estimator for anomaly classification                           |
| `MAYDAY_TIMEOUT_S` | `30.0` | Per-call timeout in seconds for cloud escalation via `MaydayAgent` (increase if using a hosted LLM with higher latency) |
| `CLOUD_API_URL` | `http://localhost:8001` | URL of the cloud deliberative planner. Set on the **edge** service so `MaydayAgent` knows where to escalate.         |

---

## Cloud Service — Environment Variables

Set these on the **cloud-api** container (or process). All are optional; defaults shown.

| Variable | Default | Purpose |
|----------|---------|---------|
| `CLOUD_LLM_BACKEND` | `mock` | LLM backend: `groq`, `anthropic`, `openrouter`, `grok`, or `mock` |
| `CLOUD_GROQ_API_KEY` | — | API key for Groq (required when `CLOUD_LLM_BACKEND=groq`) |
| `CLOUD_ANTHROPIC_API_KEY` | — | API key for Anthropic (required when `CLOUD_LLM_BACKEND=anthropic`) |
| `CLOUD_OPENROUTER_API_KEY` | — | API key for OpenRouter (required when `CLOUD_LLM_BACKEND=openrouter`) |
| `CLOUD_XAI_API_KEY` | — | API key for xAI/Grok (required when `CLOUD_LLM_BACKEND=grok`) |
| `CLOUD_EMBEDDER_BACKEND` | `mock` | Embedder for RAG: `miniLM` (sentence-transformers) or `mock` (zeros) |
| `CLOUD_VECTOR_STORE_BACKEND` | `in_memory` | Vector store: `qdrant` or `in_memory` |
| `CLOUD_QDRANT_URL` | `http://localhost:6333` | Qdrant service URL (used when `CLOUD_VECTOR_STORE_BACKEND=qdrant`) |
| `CLOUD_INCIDENT_STORE` | `sqlite` | Incident persistence: `sqlite` or `in_memory` |
| `CLOUD_DB_PATH` | `cortexguard_cloud.db` | SQLite database file path |
| `CLOUD_MIN_CONFIDENCE` | `0.5` | Minimum LLM confidence score to accept a candidate plan; plans below this threshold are rejected as `needs_human` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | OpenTelemetry OTLP HTTP endpoint (e.g. `http://tempo:4318`). Unset disables tracing. |
| `CLOUD_MAYDAY_RATE_LIMIT` | `10/minute` | Rate limit for `POST /api/v1/mayday` per client IP (slowapi format) |
| `CLOUD_RESULT_RATE_LIMIT` | `60/minute` | Rate limit for `GET /api/v1/mayday/{trace_id}/result` per client IP |
| `CLOUD_OUTCOME_RATE_LIMIT` | `30/minute` | Rate limit for `POST /api/v1/outcomes` per client IP |
| `CLOUD_LLM_TIMEOUT_S` | `20.0` | Per-call timeout in seconds for outbound LLM requests; exceeded calls route to `needs_human` |
| `CLOUD_LLM_MAX_CONCURRENCY` | `4` | Maximum number of concurrent in-flight LLM calls; additional calls queue behind the semaphore |
| `CLOUD_LLM_MAX_RETRIES` | `2` | Maximum retry attempts for retryable LLM errors (HTTP 429, 5xx) before routing to `needs_human` |
| `CLOUD_LLM_BASE_BACKOFF_MS` | `500` | Base backoff in milliseconds for LLM retry delays; actual delay uses full-jitter exponential backoff |

### Recommended production configuration

```bash
CLOUD_LLM_BACKEND=groq
CLOUD_GROQ_API_KEY=<your-key>
CLOUD_EMBEDDER_BACKEND=miniLM
CLOUD_VECTOR_STORE_BACKEND=qdrant
CLOUD_QDRANT_URL=http://qdrant:6333
CLOUD_INCIDENT_STORE=sqlite
CLOUD_DB_PATH=/data/cortexguard_cloud.db
CLOUD_MIN_CONFIDENCE=0.5
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4318
CLOUD_MAYDAY_RATE_LIMIT=10/minute
CLOUD_RESULT_RATE_LIMIT=60/minute
CLOUD_OUTCOME_RATE_LIMIT=30/minute
CLOUD_LLM_TIMEOUT_S=20
CLOUD_LLM_MAX_CONCURRENCY=4
CLOUD_LLM_MAX_RETRIES=2
CLOUD_LLM_BASE_BACKOFF_MS=500
```

### Cloud Health Endpoints

```
GET /healthz/live    → {"status": "alive"}
GET /healthz/ready   → {"status": "ok"} | HTTP 503
GET /metrics         → Prometheus exposition format
```

The readiness check verifies the SQLite database and Qdrant connection (if configured).

---

## Health Endpoints

### Liveness

```
GET /healthz/live
```

Always returns HTTP 200 while the process is running. Use this for container liveness probes.

```json
{"status": "alive"}
```

### Readiness

```
GET /healthz/ready
```

Returns HTTP 200 when all subsystems are up, HTTP 503 when any are degraded. Use this for load balancer readiness probes.

```json
{
  "blackboard": true,
  "policy_engine": true,
  "estimator": true,
  "orchestrator": true
}
```

---

## API Reference

Interactive API docs are served by the edge service at runtime:

- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

---

## Grafana Dashboard

Open `http://localhost:3000` (no login required in the demo stack).

### System Health row
Live counters for active anomalies, plan queue depth, estimator confidence, and LLM circuit breaker state. These are the first things to check during an incident — a non-zero plan queue or open circuit breaker indicates the system is under load or degraded.

### HTTP RED row
Ingestion request rate, error percentage, p95 and p99 latency, and rate-limited (429) requests/s. High error % combined with high latency suggests the edge is overloaded or a sensor is sending malformed data. A non-zero 429 rate means a client is exceeding the `INGEST_RATE_LIMIT`.

### Subsystem Latency row
Per-component p95 latency timeseries: fusion, estimator, anomaly detector, policy agent, policy generation, orchestrator. Spikes here indicate where time is being spent in the pipeline.

### Flow Rates and Outcomes row
Plan completion rate, policy escalations to cloud, E-STOP counter, step outcomes. Rising E-STOP count or escalations indicate repeated unresolved anomalies.

### SLO Error Budgets row
- Detection loop SLO: anomaly detector p95 must stay under 200ms
- Plan success rate SLO: must stay above 95%
- LLM circuit health SLO: circuit breaker must remain closed
- HTTP error budget: 5xx rate must stay low

---

## Alerts

Alerts are defined in `docker/cortexguard_alerts.yml` and routed through Prometheus.

| Alert | Severity | Meaning | Response |
|-------|----------|---------|----------|
| `LLMCircuitBreakerOpen` | warning | LLM policy engine circuit breaker has tripped | Check policy agent logs; system automatically falls back to rule-based policy while tripped |
| `DetectionLoopLatencySLOBreach` | warning | Anomaly detector p95 latency exceeds 200ms | Check CPU load on the edge host; consider reducing sensor ingestion rate |
| `PlanFailureRateHigh` | critical | More than 10% of plans failing in a 5-minute window | Check StepExecutor logs; hardware controller mock may be unresponsive |
| `MaydayCloudDegraded` | warning | 2 or more consecutive cloud escalation failures | Check network connectivity and cloud agent availability |
| `PlanQueueBacklog` | warning | Plan queue exceeds 5 items for more than 1 minute | Orchestrator may be blocked on a long-running plan; check current plan logs |
| `IngestionErrorRateHigh` | critical | Ingest endpoint returning 5xx errors at more than 0.1/s for 2 minutes | Check ingestion logs; sensor may be sending invalid payloads |
| `AnomaliesNotClearing` | warning | Active anomalies persisting for more than 5 minutes | Recovery plan may be stuck or failing; check orchestrator and safety agent state |

---

## Restart and Recovery

**Graceful shutdown:** The edge service handles SIGTERM by draining the orchestrator queue and persisting the blackboard before exit. Avoid sending SIGKILL unless the process is unresponsive.

**Restart:** Start normally with `task edge:run` or `docker compose up`. If `PERSISTENCE_ENABLED=true`, the blackboard re-hydrates from the last snapshot on startup. The startup log will confirm:
- `"Blackboard state restored from snapshot"` — prior state recovered
- `"Blackboard initialized fresh"` — no snapshot found or persistence disabled

**Mid-plan crash:** If the process dies while a plan is executing, the plan will not auto-resume on restart. Anomaly detection will re-evaluate the system state on the next tick and re-trigger a remediation plan if the anomaly condition still holds.
