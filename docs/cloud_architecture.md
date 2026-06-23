# Cloud Architecture

CortexGuard's cloud tier is a FastAPI service that receives escalations from the edge `MaydayAgent` and runs a deliberative
planning workflow to generate multi-step recovery plans. It is decoupled from the edge by design: safety-critical decisions
never block on cloud availability.

---

## Deployment Architecture

Three processes share the cloud tier:

| Process | Entry point | Runs in | Responsibilities |
|---------|-------------|---------|------------------|
| **cloud-api** | `uvicorn cortexguard.cloud.runtime:app` | Fargate service | REST API (`/mayday`, `/outcomes`, `/healthz`), serves metrics |
| **worker** | `python -m cortexguard.cloud.worker` | Fargate service | Long-polls SQS, runs LangGraph planning workflow, handles resume, TTL cleanup |
| **mcp-server** | `python -m cortexguard.cloud.mcp_server` | Launched per-operator via `docker exec` into the API container, or run locally by the MCP host | MCP stdio tools for incident lookup, plan override, operator resolution |

All three share the same Postgres incident store, Qdrant vector store, and SQS queue.

## Overview

Two deployment modes:

**In-process (dev)** no SQS, no worker. The cloud-api runs the graph directly in a background task:

```
MaydayAgent (edge)
    │  POST /api/v1/mayday  (MaydayPacket)
    ▼
cloud-api FastAPI
    │  202 Accepted  →  asyncio.create_task
    ▼
CloudOrchestrator.run()
    │  runs LangGraph planning workflow
    ▼
GET /api/v1/mayday/{trace_id}/result  (polling)
    │  plan_ready | needs_human | no_safe_plan
    ▼
Edge receives Plan
```

**SQS worker (production)**  the API enqueues to SQS and a separate Fargate worker runs the graph:

```
MaydayAgent (edge)
    │  POST /api/v1/mayday
    ▼
cloud-api FastAPI
    │  202 Accepted  →  SQSCloudOrchestrator.submit()
    │                      writes "pending" IncidentRecord + enqueue to SQS
    ▼
SQS Queue
    │  long-poll
    ▼
Worker (Fargate task, python -m cortexguard.cloud.worker)
    │  CloudOrchestrator.run_once(packet)
    ▼
LangGraph workflow (see below)
    │
    ├── plan_ready / no_safe_plan  →  finalise IncidentRecord in Postgres
    └── interrupted (needs_human)  →  checkpoint saved, await operator via MCP
```

The edge polls `GET /mayday/{trace_id}/result` until the decision is persisted. The SQS mode decouples ingestion throughput from LLM latency and allows independent scaling of API and worker tasks.

---

## LangGraph Planning Workflow

The planner is a compiled LangGraph `StateGraph` with six nodes. All nodes operate on a shared `CloudPlanningState` typed dict. Each node receives the full state and returns a partial dict of the fields it changed; LangGraph merges the returned dict into the shared state before passing to the next node.

```python
class CloudPlanningState(TypedDict):
    request: NotRequired[MaydayPacket]
    incident_id: str | None
    retrieved_incidents: list[dict]
    retrieved_incident_records: list[IncidentRecord]
    candidate_plan: Plan | None
    validation_result: ValidationResult | None
    decision: str | None
    rationale: str | None
    confidence: float | None
    needs_human_review: bool
    errors: list[str]
    operator_response: NotRequired[dict[str, Any] | None]


class ResumeInput(TypedDict, total=False):
    """Partial state passed on resume,  only operator_response is set.
    LangGraph merges this with the checkpointed state."""

    operator_response: dict[str, Any] | None
```

```
persist_incident
    │
retrieve_similar_incidents   ← Qdrant vector search (RAG)
    │
generate_candidate_plan      ← LLM structured output
    │
validate_candidate_plan      ← capability + confidence checks
    │
route_decision
    │  ─ conditional ──┬── END  (plan_ready / no_safe_plan / needs_human w/o review flag)
    │                  │
    │                  └── pause_for_operator  (needs_human + needs_human_review=True)
    │                                          └─ interrupt() → checkpoint → await operator
```

### Node 1: `persist_incident`

Writes an `IncidentRecord` to the configured incident store (Postgres in production, SQLite in dev) immediately on receiving the `MaydayPacket`. This ensures every escalation is durably recorded even if subsequent nodes fail. Fields: `device_id`, `anomaly_key`, `severity`, `summary`, `raw_packet_json`, `decision=pending`.

### Node 2: `retrieve_similar_incidents`

Embeds a summary of the incoming `MaydayPacket` using `MiniLMEmbedder` (`all-MiniLM-L6-v2`, 384-dim, CPU) and performs an approximate nearest-neighbour search in Qdrant.

- **Filter**: if the packet has a single anomaly key, the search is pre-filtered to only return incidents with the same `anomaly_key`.
- **Learning-to-rank re-ranking**: results are re-scored by a composite function (`ranker.py`): `composite = similarity + outcome_boost - failure_penalty`. `resolved` outcomes add `CLOUD_RETRIEVAL_OUTCOME_BOOST` (default `0.2`); high-confidence `plan_ready` adds half that; unresolved `needs_human`/`no_safe_plan` subtract `CLOUD_RETRIEVAL_FAILURE_PENALTY` (default `0.1`).
- Returns up to 5 similar `IncidentRecord`s, injected into the LLM prompt as context. Similarity scores are persisted on the incident record.

### Node 3: `generate_candidate_plan`

Calls the configured `LLMClientProtocol` with a `PlannerRequest` containing the escalation summary, sensor state, retrieved incident summaries, device capability catalog, and anomaly details. Uses `instructor` for structured output extraction.

Returns a `PlannerResponse` with:
- `candidate_plan: Plan | None`:  multi-step plan using only registered capabilities
- `confidence: float`:  self-assessed confidence [0.0-1.0]
- `rationale: str`:  plain-language explanation
- `needs_human_review: bool`:  LLM's flag that operator review is needed
- `raw_provider_metadata: dict`:  LLM-specific metadata

After the LLM returns, `_normalise_plan()` forces correct provenance: `source=PlanSource.CLOUD_AGENT`, `trace_id` set to the packet's trace_id, and any non-UUID IDs regenerated.

### Node 4: `validate_candidate_plan`

`PlanValidator` checks the candidate plan against three criteria:

1. **Capability validation**: every `PlanStep.action` must exist in the `CapabilityRegistry`.
2. **Confidence threshold**: if `confidence < CLOUD_MIN_CONFIDENCE` (default `0.5`), the plan is rejected.
3. **Human review flag**: if `needs_human_review=True`, the plan is rejected regardless of confidence.

Failed validation increments `cloud_validation_failures_total`. Returns a `ValidationResult` with `passed`, `errors`, and `risk_level`.

### Node 5: `route_decision`

Determines the final decision string:

| Condition | Decision |
|-----------|----------|
| Any error in `state["errors"]` | `needs_human` |
| `candidate_plan is None` | `no_safe_plan` |
| `validation_result.passed is False` | `needs_human` |
| Otherwise | `plan_ready` |

Increments `cloud_decisions_total{decision=...}`.

A conditional edge (`route_after_decision`) then decides the next node:

- If `decision == "needs_human"` **and** `needs_human_review == True` → `"pause_for_operator"`
- Otherwise → `END` (graph finishes)

This distinguishes between **LLM-requested human review** (pause for operator) and **validation failures / errors** (graph finishes, operator inspects via MCP).

### Node 6: `pause_for_operator`

Only reached when the LLM explicitly requested human review. On **first entry** (no `operator_response` yet):

1. Logs the event and increments `cloud_needs_human_total`
2. Calls `interrupt({...})`: LangGraph's checkpoint primitive that saves the full graph state to the checkpointer and yields control back to the caller

On **resume** (when `operator_response` is present from a previous checkpoint):

| `operator_response` | Result |
|---------------------|--------|
| `{"timeout": True}` | `no_safe_plan` (auto-resolve) |
| `{"approved": True}` | `plan_ready` (optionally with `plan_override`) |
| `{"action": "reject"}` | `no_safe_plan` |
| otherwise | re-enters `interrupt()` |

---

## Checkpointer (LangGraph Persistence)

The checkpointer is a `BaseCheckpointSaver[Any]` instance that saves interrupted graph state for later resumption. Created by `create_checkpointer()`:

| Config | Class | Where |
|--------|-------|-------|
| `"memory"` (default) | `MemorySaver` | in-process RAM only |
| `"sqlite"` | `AsyncSqliteSaver` | local `checkpoints.db` file |
| `"postgres"` (production) | `PostgresSaver` | `checkpoints` table in Postgres |

Set via `CLOUD_CHECKPOINT_STORE` and optionally `CLOUD_CHECKPOINT_DB_PATH`. In production ECS, the worker passes a `PostgresSaver` to `CloudOrchestrator`, which passes it to `graph.compile(checkpointer=...)`.

---

## Operator Resolution Flow

When the graph is paused at `pause_for_operator`:

1. The worker detects `interrupted=True` after `run_once()`, deletes the SQS message, and returns
2. The operator interacts via **MCP** tools (the `cortexguard` MCP server running alongside the API)
3. The operator calls `record_operator_resolution` with `incident_id`, `actions_taken`, `outcome`, and optional `notes`
4. The MCP handler:
   - Persists the resolution to the incident record
   - Re-embeds the enriched summary into Qdrant (feedback loop for RAG)
   - **Pushes an SQS message** with `action="resume"` and the `operator_response` dict
5. The worker picks up the resume message, calls `orchestrator.resume(thread_id, operator_response)`, which reads the checkpoint from Postgres and calls `graph.ainvoke` with the operator input
6. The graph wakes up in `pause_for_operator`, processes the response, and finalises the incident

### Stale Checkpoint Cleanup

The worker runs a background TTL task (`_ttl_cleanup`, every 5 min) that auto-resolves any checkpoints older than 1 hour with `{"timeout": True}`, producing a `no_safe_plan` decision. This prevents interrupted graphs from hanging indefinitely if the operator never responds.

---

## RAG Pipeline

```
MaydayPacket
    │  embed summary text
    ▼
MiniLMEmbedder  →  384-dim float vector
    │
    ▼
Qdrant.search(vector, top_k=5, filter={anomaly_key})
    │  SearchResult list with scores
    ▼
Composite re-ranker: similarity + outcome_boost - failure_penalty
    │
    ▼
top-5 IncidentRecords  →  summaries injected into LLM prompt
```

Qdrant is pre-seeded on startup from `src/cortexguard/cloud/data/seeds/*.json` via `SeedLoader`. The seeder skips if the collection already contains data.

---

## LLM Backend Factory

Selected at startup by `CLOUD_LLM_BACKEND` via `get_llm_client(backend)` in `factory.py`.

| Backend | Class | API                                                   |
|---------|-------|-------------------------------------------------------|
| `groq` (default) | `GroqLLMClient` | `https://api.groq.com/openai/v1`  (OpenAI-compatible) |
| `anthropic` | `AnthropicLLMClient` | Anthropic SDK + `instructor`                          |
| `openrouter` | `OpenRouterLLMClient` | `https://openrouter.ai/api/v1`  (OpenAI-compatible)   |
| `grok` | `GrokLLMClient` | `https://api.x.ai/v1`  (OpenAI-compatible)            |
| `mock` | `MockLLMClient` | Deterministic canned response (no API key needed)     |

All backends use `instructor` for structured output extraction to a `PlannerResponse` Pydantic model. Rate limiting is handled by `LLMThrottler` (configurable concurrency, timeout, retries with backoff).

---

## Incident Persistence

Every escalation is written to the configured incident store (SQLite by default, or Postgres when `CLOUD_INCIDENT_STORE=postgres`). The schema:

| Column | Type | Description |
|--------|------|-------------|
| `incident_id` | TEXT (UUID) | Primary key |
| `escalation_id` | TEXT | Edge trace_id |
| `trace_id` | TEXT | Duplicate of escalation_id for convenience |
| `device_id` | TEXT | Edge device identifier |
| `anomaly_key` | TEXT | Primary anomaly key |
| `anomaly_type` | TEXT | Classification tag |
| `severity` | TEXT | `low`, `medium`, `high`, `critical` |
| `summary` | TEXT | Human-readable summary (also used for RAG embedding) |
| `raw_packet_json` | TEXT | Full `MaydayPacket` JSON |
| `retrieved_incident_ids_json` | TEXT | JSON list of Qdrant neighbour IDs used |
| `retrieved_incidents_json` | TEXT | `[{incident_id, similarity_score}]` from RAG with re-ranking |
| `candidate_plan_json` | TEXT | Serialised `Plan` (null if not generated) |
| `validation_errors_json` | TEXT | JSON list of validation error strings |
| `decision` | TEXT | `plan_ready`, `needs_human`, `no_safe_plan`, or `pending` |
| `rationale` | TEXT | LLM rationale string |
| `confidence` | REAL | LLM confidence score [0.0-1.0] |
| `created_at` | TEXT | ISO 8601 UTC timestamp |

Outcomes can be reported back by the edge via `POST /api/v1/outcomes` and queried via `GET /api/v1/outcomes/recent`.

---

## Step Telemetry Store

Step outcomes and sensor snapshots are sent by the edge `TelemetryClient` and ingested via `POST /api/v1/telemetry`. The store shares the same backend as incidents (SQLite by default, or Postgres when `CLOUD_INCIDENT_STORE=postgres`):

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK) | Auto-increment |
| `device_id` | TEXT | Edge device identifier |
| `key` | TEXT | Step ID |
| `outcome` | TEXT | `completed`, `retry_exhausted`, or `aborted` |
| `recorded_at` | TEXT | ISO 8601 UTC timestamp from the edge |
| `ingested_at` | TEXT | ISO 8601 UTC timestamp on ingestion |
| `sensor_snapshot_json` | TEXT | Serialised `FusionSnapshot` at step outcome time |

Indexed on `(device_id, recorded_at)` for efficient per-device queries. The weekly SageMaker retraining pipeline reads from this table to build training datasets.

---

## Step Classification

The cloud proxies classification requests from the edge to a SageMaker endpoint running a trained model. The `StepClassifierClient` on the edge calls `POST /api/v1/classify` instead of the local `MockStepClassifier` when `CLOUD_API_URL` is configured.

The classify endpoint (`src/cortexguard/cloud/classify/api.py`) runs the SageMaker call in a thread pool to avoid blocking the event loop:

| Field | Type | Description |
|-------|------|-------------|
| `predicted_outcome` | string | `completed` or `failed` |
| `confidence` | float | Model confidence [0.0-1.0] |
| `model_id` | string | Model identifier from registry |
| `model_version` | string/int | Version from the model registry |

The SageMaker endpoint name is set via `SAGEMAKER_ENDPOINT_NAME` env var (injected by Terraform in production). If unset, the endpoint returns HTTP 503. Rate limited to 100 requests/minute.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/mayday` | Receive a `MaydayPacket`, enqueue for planning, return `trace_id` |
| `GET` | `/api/v1/mayday/{trace_id}/result` | Poll for planning result (`pending`, `plan_ready`, `needs_human`, `no_safe_plan`) |
| `POST` | `/api/v1/outcomes` | Report execution outcome (edge → cloud feedback loop) |
| `GET` | `/api/v1/outcomes/recent` | List recent outcomes |
| `POST` | `/api/v1/telemetry` | Ingest step-telemetry records (sensor snapshot + outcome) from edge |
| `GET` | `/api/v1/telemetry/recent` | List recent telemetry records for debugging |
| `POST` | `/api/v1/classify` | Classify step outcome via SageMaker endpoint (proxied from edge) |
| `GET` | `/healthz/live` | Liveness probe |
| `GET` | `/healthz/ready` | Readiness probe (checks DB + Qdrant) |
| `GET` | `/metrics` | Prometheus metrics |

Interactive API docs are served at `http://localhost:8001/docs` (Swagger UI).

---

## MCP Tools (Operator Interface)

The MCP server (`python -m cortexguard.cloud.mcp_server`) exposes tools for operator interaction:

| Tool | Purpose |
|------|---------|
| `lookup_incident` | Get incident details by trace_id or incident_id |
| `lookup_outcome` | Get operator resolution for an incident |
| `get_latest_planner_decision` | View the most recent planning result |
| `replan_incident` | Force re-planning for an incident with a new prompt |
| `update_incident_candidate_plan` | Manually override the candidate plan |
| `record_operator_resolution` | Record operator decision and resume a paused graph |

The operator connects via `claude mcp add cortexguard -- <command>` or any MCP-compatible client.

---

## Worker Process

The worker (`python -m cortexguard.cloud.worker`) is a long-running Fargate task that:

1. Long-polls SQS for planning requests and resume messages
2. Runs `CloudOrchestrator.run_once(packet)` for new planning requests
3. Detects graph interruptions (needs_human) and deletes the message, the checkpoint is the source of truth
4. Handles `action="resume"` messages by calling `CloudOrchestrator.resume()` with the operator response
5. Runs a background TTL cleanup task that auto-resolves stale checkpoints after 1 hour

The worker requires `CLOUD_INCIDENT_STORE=postgres` and `CLOUD_SQS_QUEUE_URL` to be set.

---

## Docker Compose

- `docker-compose.cloud.yml`: standalone cloud stack (cloud-api + Qdrant)
- `docker-compose.demo.yaml`: full demo stack (edge + simulator + cloud-api + Qdrant + Prometheus + Grafana + Tempo)

```bash
# Standalone cloud stack (in-process mode)
CLOUD_GROQ_API_KEY=<key> docker compose -f docker-compose.cloud.yml up --build

# Full demo (edge + cloud)
CLOUD_GROQ_API_KEY=<key> docker compose -f docker-compose.demo.yaml up --build
```

---

## See Also

- `docs/agents_overview.md`: agent roles and responsibilities
- `docs/observability.md`: cloud metrics and traces
- `docs/OPERATIONS.md`: environment variable reference
- `src/cortexguard/cloud/graph/workflow.py`: LangGraph graph construction
- `src/cortexguard/cloud/graph/nodes.py`: individual node implementations
- `src/cortexguard/cloud/orchestrator.py`: CloudOrchestrator, SQSCloudOrchestrator, resume flow
- `src/cortexguard/cloud/worker.py`: SQS worker with resume and TTL cleanup
- `src/cortexguard/cloud/mcp_server.py`: MCP tools for operator resolution
- `src/cortexguard/cloud/planner/factory.py`: LLM backend factory
- `src/cortexguard/cloud/retrieval/store.py`: RAG retrieval with outcome boosting
- `src/cortexguard/cloud/validation/plan_validator.py`: capability and confidence validation
- `docs/ml_infrastructure.md`: SageMaker retraining pipeline, model registry, data drift, and step classifier endpoint
