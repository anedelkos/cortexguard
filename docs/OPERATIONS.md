# CortexGuard Edge — Operations Reference

Audience: deploying, configuring, or debugging the edge service.

---

## Environment Variables

All variables are optional with the defaults shown. Set them in your shell, `.env` file, or Docker Compose `environment:` block.

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEVICE_ID` | `mock_01` | Device identity tag in logs and traces |
| `RUNTIME_PROFILE` | `default` | Runtime profile selector |
| `POLICY_MODEL_ID` | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model ID for LLM policy engine |
| `POLICY_USE_MOCK` | `true` | `false` to enable real Mistral-7B inference (requires full deps + GPU recommended) |
| `POLICY_REMEDIATION_COOLDOWN_S` | `30.0` | Minimum seconds between remediation policy generations for the same anomaly |
| `PERSISTENCE_ENABLED` | `false` | Enable periodic blackboard snapshots to disk |
| `PERSISTENCE_FILE_PATH` | `/var/lib/cortexguard/blackboard.json` | Blackboard snapshot location |
| `PERSISTENCE_SNAPSHOT_INTERVAL` | `5.0` | Seconds between snapshots |
| `OTLP_ENDPOINT` | `http://tempo:4318/v1/traces` | OpenTelemetry trace collector endpoint |
| `LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_JSON` | `true` | JSON structured logs in prod; set `false` for readable local output |
| `INGEST_RATE_LIMIT` | `100/second` | Rate limit for `POST /api/v1/ingest` per client IP (slowapi format, e.g. `200/second`, `1000/minute`) |
| `FUSION_FORCE_MIN_N` | `0.0` | Force sensor floor in Newtons (values below are clamped) |
| `FUSION_FORCE_DROP_PCT` | `100.1` | Force drop detection threshold as % change |
| `FUSION_DRIFT_FAIL_MM` | `10.0` | Position drift failure threshold in mm |

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
