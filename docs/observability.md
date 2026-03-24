# CortexGuard Observability

CortexGuard is a real-time safety-critical system. When something goes wrong — a detection loop slows down, an LLM policy fails, a cloud escalation times out — we need to know immediately and understand why. This document describes what the observability layer tracks, why each signal matters, and how to read the system's health at a glance.

---

## Three Pillars

CortexGuard uses three complementary observability mechanisms:

| Pillar | Tool | Purpose |
|--------|------|---------|
| **Metrics** | Prometheus + Grafana | Numerical time-series for alerting, dashboards, and SLOs |
| **Traces** | OpenTelemetry → Tempo | Request-scoped spans for debugging specific events |
| **Logs** | stdout (JSON in prod) | Human-readable narrative of what happened and why |

These three are complementary, not redundant. Metrics tell you *that* something is wrong. Traces tell you *where*. Logs tell you *why*.

---

## Metrics

Metrics are scraped by Prometheus every 5 seconds from `http://edge:8080/metrics`. They are the primary signal for alerting and dashboards.

All metrics follow a low-cardinality design: no labels contain dynamic values like anomaly IDs, plan IDs, or trace IDs. This keeps Prometheus memory usage bounded and queries fast.

### Latency — `cortexguard_component_duration_ms`

A single histogram with a `component` label covering every major subsystem:

| Component | What it measures | Why it matters |
|-----------|-----------------|----------------|
| `fusion_process_record` | Sensor fusion pipeline per record | Upstream of everything — if this is slow, all downstream stages are starved |
| `estimator_update` | Online learner state estimation | Slow updates mean stale confidence and degraded anomaly detection quality |
| `anomaly_detector_tick` | Full detector ensemble tick | Direct measure of detection loop health; budget is 50–200ms |
| `policy_agent_tick` | Policy agent control loop | Measures overall policy responsiveness including LLM dispatch overhead |
| `policy_generation` | LLM inference time | The most variable and expensive step; 30s timeout means a spike here blocks remediation |
| `policy_publish` | Publishing a policy to the blackboard | Should be near-zero; spikes indicate blackboard contention |
| `orchestrator_tick` | Orchestrator control loop | Measures plan queue throughput and step advancement |
| `mayday_attempt` | Single cloud escalation HTTP attempt | Direct measure of cloud availability from the edge's perspective |
| `mayday` | Full escalation including retries | End-to-end escalation latency including backoff |

Use `histogram_quantile(0.95, ...)` to get p95 latency — the mean hides tail behavior that matters in a safety system.

### HTTP Ingestion — Rate, Errors, Duration

| Metric | Type | What it measures |
|--------|------|-----------------|
| `cortexguard_http_requests_total` | Counter | HTTP requests by `method` and `status_code` (202, 422, 500) |
| `cortexguard_http_request_duration_ms` | Histogram | End-to-end ingestion handler latency in milliseconds |

Use `rate(cortexguard_http_requests_total{status_code=~"4..|5.."}[5m])` for error rate and `histogram_quantile(0.95, rate(cortexguard_http_request_duration_ms_bucket[5m]))` for p95 latency.

### Safety State — Gauges

| Metric | What it shows | Alert threshold |
|--------|--------------|-----------------|
| `cortexguard_active_anomalies` | Count of anomalies currently active (not yet resolved) | Any value > 0 sustained for > 5 minutes |
| `cortexguard_plan_queue_length` | Plans waiting to be executed by the orchestrator | > 5 for more than 1 minute suggests the executor is falling behind |
| `cortexguard_estimator_confidence` | State estimator's confidence in its model [0.0–1.0] | Trending toward 0 means sensor data is diverging from the learned model |
| `cortexguard_llm_circuit_open` | LLM circuit breaker state: 1 = open (tripped), 0 = closed | Any value of 1 means unknown/medium anomalies receive fallback policy |
| `cortexguard_mayday_consecutive_failures` | Current streak of consecutive cloud escalation failures | ≥ 2 indicates cloud connectivity is degraded |

The system health row of the Grafana dashboard includes all five gauges. They answer the question: *is the system currently in a normal state?*

### Event Counts — Counters

Counters only go up. Use `rate(...)` in Prometheus to see events per second over a window.

| Metric | What it counts | Why it matters |
|--------|---------------|----------------|
| `cortexguard_anomalies_total` | Anomalies emitted by the detection ensemble | Baseline rate tells you what normal looks like; spikes indicate incidents |
| `cortexguard_detector_failures_total` | Exceptions thrown by individual sub-detectors | Any non-zero rate means a detector is crashing — its signal is missing from the ensemble |
| `cortexguard_policy_escalations_total` | Anomalies that required Mayday cloud escalation | High rate means on-device policy is insufficient; zero for long periods might indicate escalation is broken |
| `cortexguard_llm_requests_total` | LLM policy generation calls by outcome (success, failure, timeout, circuit_skipped) | Distinguishes LLM failures from circuit-breaker skips |
| `cortexguard_mayday_escalations_total` | Cloud escalation attempts by outcome (success, timeout, error, exhausted) | Exposes MaydayAgent's internal retry counters as Prometheus metrics |
| `cortexguard_plans_total` | Plans reaching a terminal state, by status (completed, failed) | Enables plan success rate SLOs |
| `cortexguard_steps_total` | Step executions by outcome (completed, retry_exhausted, aborted) | Surfaces step-level failure and retry exhaustion rates |

---

## Traces

CortexGuard instruments every significant agent operation with OpenTelemetry spans. Spans are exported via OTLP HTTP to Tempo and queryable in Grafana's Explore view.

### What gets traced

Each subsystem creates a named tracer and wraps its key operations:

| Tracer | Key spans | Notable attributes |
|--------|-----------|-------------------|
| `cortexguard.edge_fusion` | `process_record` | `window.size`, `has_rgb` |
| `cortexguard.online_learner_state_estimator` | `estimator_update` | `feature.count` |
| `cortexguard.anomaly_detector` | `anomaly_detector_tick`, `detector_failure` | `detector.name`, `anomaly.key`, `anomaly.severity` |
| `cortexguard.policy_agent` | `policy_agent_tick`, `policy_generation`, `escalation_triggered` | `policy.id`, `llm.duration_ms`, `model` |
| `cortexguard.orchestrator` | `orchestrator_tick`, `plan_queued`, `plan_completed` | `active_anomalies.count` |
| `cortexguard.mayday` | `mayday_attempt`, `cloud_call_failed` | — |

### When to use traces vs metrics

Use metrics when you want to know *whether* a problem exists (dashboards, alerts, rate-of-change). Use traces when you want to understand *what happened during a specific event* — for example, tracing a particular anomaly through detection → policy generation → plan execution to see where time was spent or where an error occurred.

Traces also carry the `ReasoningTraceEntry` log — a structured human-readable narrative of the agent's reasoning (e.g. "POLICY_LLM_ERROR: model timed out after 30s, falling back to safe policy"). This reasoning trace is the primary debugging tool for understanding why the system made a specific decision.

---

## Logs

Logs are written to stdout. In production (Docker), `LOG_JSON=true` emits structured JSON. In development, plain text is used.

Log level is controlled by `LOG_LEVEL` (default: `INFO`). Set to `DEBUG` for verbose subsystem output including per-tick details.

Logs complement traces by providing a lower-level narrative. If a trace shows an anomaly was detected but no policy was generated, the logs will show whether the LLM circuit breaker was open, which fallback was used, and what the exact error message was.

---

## Grafana Dashboard

The dashboard ("CortexGuard Edge Observability", UID `cortexguard-edge-observability`) is provisioned automatically from `docker/grafana_dashboard.json`. It refreshes every 5 seconds and defaults to a 15-minute time window.

### Row 1 — System Health

Six stat and time series panels: the three original safety-state gauges (active anomalies, plan queue length, estimator confidence) plus three new panels — LLM circuit breaker state (green=closed, red=open), mayday consecutive failures, and ingestion rate. These are the first thing to check when something looks wrong.

### Row 2 — HTTP RED (collapsed)

Four panels providing full RED method coverage for the HTTP ingestion endpoint: request rate by status code, error rate percentage (4xx+5xx), p95 latency stat, and p99 latency time series.

### Row 3 — Subsystem Latency Percentiles

Six time series panels (replacing the previous heatmaps) showing p50/p95/p99 latency lines for each major subsystem: fusion, estimator, anomaly detector, policy agent, policy generation, and orchestrator. Reference lines indicate the latency budget thresholds (200ms for detection, 300ms for fusion and decision).

### Row 4 — Flow Rates and Outcomes

Eight panels: the three original counters (anomalies, detector failures, escalations) plus five new panels — plan completion rate by status, plan success ratio, LLM request rate by outcome, mayday escalation outcomes, and step outcome rate.

### Row 5 — SLO Error Budgets (collapsed)

Four stat panels tracking key SLOs: detection loop p95 latency vs 200ms budget, plan success rate vs 95% target, LLM circuit health, and HTTP error budget remaining. All panels use green/yellow/red thresholds at the 99%/95% boundaries.

---

## Design Principles

**Safety-first latency budgets.** The detection and fusion loops have hard latency budgets (50–200ms, 100–300ms). Metrics are designed to make budget violations visible immediately via p95/p99 histogram quantiles, not just averages.

**Low cardinality.** No metric label contains a per-event identifier (anomaly ID, plan ID, trace ID). Labels are bounded enumerations. This keeps Prometheus memory usage predictable regardless of event volume.

**Traces for decisions, metrics for trends.** The LLM's reasoning, the specific anomaly that triggered escalation, the exact step that failed — these belong in traces where they carry full context. Aggregated counts and latencies belong in metrics where they can be charted over time.

**Defense in depth.** The system has multiple internal fault-tolerance mechanisms: an LLM circuit breaker (trips after 3 consecutive failures, 60s cooldown), MaydayAgent retry logic (2 attempts, exponential backoff), and step-level retries (3 attempts per step). Observability surfaces all of these so that degraded-but-operational states are visible, not silent.

---

## See Also

- `docs/metrics_spec.md` — full metric definitions, label constraints, and cardinality rules
- `docs/architecture.md` — system architecture and latency budget context
- `src/cortexguard/edge/utils/metrics.py` — metric definitions
- `src/cortexguard/edge/utils/tracing.py` — trace sink and span helpers
- `docker/grafana_dashboard.json` — dashboard panel definitions
- `docker/prometheus.yml` — scrape configuration
- `docker/cortexguard_alerts.yml` — Prometheus alerting rules
