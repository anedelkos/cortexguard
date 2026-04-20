# Cloud Architecture

CortexGuard's cloud tier is a FastAPI service that receives escalations from the edge `MaydayAgent` and runs a deliberative
planning workflow to generate multi-step recovery plans. It is decoupled from the edge by design: safety-critical decisions
never block on cloud availability.

---

## Overview

```
MaydayAgent (edge)
    │  POST /api/v1/mayday  (MaydayPacket)
    ▼
cloud-api FastAPI
    │  202 Accepted  →  background task
    ▼
CloudOrchestrator
    │  runs LangGraph planning workflow
    ▼
GET /api/v1/mayday/{trace_id}/result  (polling)
    │  plan_ready | needs_human | no_safe_plan
    ▼
MaydayAgent receives Plan, submits to edge Orchestrator
```

The edge polls the result endpoint until a decision is available or `MAYDAY_TIMEOUT_S` expires.

---

## LangGraph Planning Workflow

The planner is a compiled LangGraph `StateGraph` with five sequential nodes. All nodes operate on a shared `CloudPlanningState` typed dict.

```
persist_incident
    │
retrieve_similar_incidents   ← Qdrant vector search (RAG)
    │
generate_candidate_plan      ← LLM structured output
    │
validate_candidate_plan      ← capability + confidence checks
    │
route_decision               → plan_ready | needs_human | no_safe_plan
```

### Node 1 — `persist_incident`

Writes an `IncidentRecord` to SQLite immediately on receiving the `MaydayPacket`. This ensures every escalation is durably
recorded even if subsequent nodes fail. Fields: `device_id`, `anomaly_key`, `severity`, `summary`, `raw_packet_json`, `decision=pending`.

### Node 2 — `retrieve_similar_incidents`

Embeds a summary of the incoming `MaydayPacket` (`device_id + anomaly_keys + plan_id`) using `MiniLMEmbedder` (`all-MiniLM-L6-v2`, 384-dim, CPU)
and performs an approximate nearest-neighbour search in Qdrant.

- **Filter**: if the packet has a single anomaly key, the search is pre-filtered to only return incidents with the same `anomaly_key`. This prevents cross-anomaly noise.
- **Outcome boost**: incidents whose stored `decision` is `plan_ready` receive a score multiplier of `1.1x` — previously successful recoveries are ranked higher as priors.
- Returns up to 5 similar `IncidentRecord`s. Their summaries are injected into the LLM prompt as context.

### Node 3 — `generate_candidate_plan`

Calls the configured `LLMClientProtocol` implementation with a `PlannerRequest` containing:

- `escalation_summary`: device ID and anomaly key
- `state_summary`: JSON-encoded state estimate from the edge
- `retrieved_summaries`: summaries of the top similar past incidents
- `capability_catalog_json`: the device's registered capabilities (loaded from `capability_registry.yaml`)
- `anomaly_key` and `severity`

The LLM is instructed (via `instructor` structured output extraction) to return a `PlannerResponse` Pydantic model with:

- `candidate_plan: Plan | None` — a full `Plan` with `PlanStep`s using only capabilities from the catalog
- `confidence: float` — the model's self-assessed confidence [0.0–1.0]
- `rationale: str` — plain-language explanation of the plan
- `needs_human_review: bool` — whether the model flagged operator review
- `raw_provider_metadata: dict` — pass-through for any LLM-specific metadata

**Post-generation normalisation**: After the LLM returns, `_normalise_plan()` forces correct provenance — `source=PlanSource.CLOUD_AGENT`, `trace_id` set to the packet's trace_id, and any non-UUID `plan_id` / step `id` values regenerated. This ensures the edge can always deserialise the plan regardless of what the LLM chose to emit.

### Node 4 — `validate_candidate_plan`

`PlanValidator` checks the candidate plan against two criteria:

1. **Capability validation** (`CapabilityAdapter`): every `PlanStep.action` must exist in the `CapabilityRegistry` loaded from `src/cortexguard/common/capability_registry.yaml`. Steps referencing unknown actions fail validation.
2. **Confidence threshold**: if `confidence < CLOUD_MIN_CONFIDENCE` (default `0.5`), the plan is rejected — a low-confidence plan is riskier than `needs_human`.
3. **Human review flag**: if `needs_human_review=True` was returned by the LLM, the plan is rejected regardless of confidence.

Failed validation increments `cloud_validation_failures_total`. The `ValidationResult` carries `passed`, `errors`, and `risk_level`.

### Node 5 — `route_decision`

Determines the final decision string based on state:

| Condition | Decision |
|-----------|----------|
| Any error in `state["errors"]` | `needs_human` |
| `candidate_plan is None` | `no_safe_plan` |
| `validation_result.passed is False` | `needs_human` |
| Otherwise | `plan_ready` |

Increments `cloud_decisions_total{decision=...}` and `cloud_needs_human_total` (when applicable).

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
outcome-boost: plan_ready incidents × 1.1
    │
    ▼
top-5 IncidentRecords  →  summaries injected into LLM prompt
```

Qdrant is pre-seeded on startup from `src/cortexguard/cloud/data/seeds/*.json` via `SeedLoader`. Each seed file contains one past incident with a worked example plan. The seeder skips seeding if the collection already contains data.

---

## LLM Backend Factory

The backend is selected at startup by `CLOUD_LLM_BACKEND` via `get_llm_client(backend)` in `src/cortexguard/cloud/planner/factory.py`.

| Backend | Class | API |
|---------|-------|-----|
| `groq` (default) | `GroqLLMClient` | `https://api.groq.com/openai/v1` — OpenAI-compatible |
| `anthropic` | `AnthropicLLMClient` | Anthropic SDK + `instructor` |
| `openrouter` | `OpenRouterLLMClient` | `https://openrouter.ai/api/v1` — OpenAI-compatible |
| `grok` | `GrokLLMClient` | `https://api.x.ai/v1` — OpenAI-compatible |
| `mock` | `MockLLMClient` | Deterministic canned response (no API key needed) |

Groq and the OpenAI-compatible backends use `instructor.from_openai(AsyncOpenAI(...))` for structured output extraction. The Anthropic backend uses `instructor.from_anthropic(AsyncAnthropic(...))`.

---

## Incident Persistence

Every escalation is written to SQLite (`CLOUD_DB_PATH`). The schema is:

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
| `candidate_plan_json` | TEXT | Serialised `Plan` (null if not generated) |
| `validation_errors_json` | TEXT | JSON list of validation error strings |
| `decision` | TEXT | `plan_ready`, `needs_human`, `no_safe_plan`, or `pending` |
| `rationale` | TEXT | LLM rationale string |
| `confidence` | REAL | LLM confidence score [0.0–1.0] |
| `created_at` | TEXT | ISO 8601 UTC timestamp |

Outcomes can be reported back by the edge via `POST /api/v1/outcomes` and queried via `GET /api/v1/outcomes/recent`.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/mayday` | Receive a `MaydayPacket`, start planning workflow, return `trace_id` |
| `GET` | `/api/v1/mayday/{trace_id}/result` | Poll for planning result (`pending`, `plan_ready`, `needs_human`, `no_safe_plan`) |
| `POST` | `/api/v1/outcomes` | Report execution outcome (edge → cloud feedback loop) |
| `GET` | `/api/v1/outcomes/recent` | List recent outcomes |
| `GET` | `/healthz/live` | Liveness probe |
| `GET` | `/healthz/ready` | Readiness probe (checks DB + Qdrant) |
| `GET` | `/metrics` | Prometheus metrics |

Interactive API docs are served at `http://localhost:8001/docs` (Swagger UI).

---

## Docker Compose

The cloud service is available in two compose files:

- `docker-compose.cloud.yml` — standalone cloud stack (cloud-api + Qdrant)
- `docker-compose.demo.yaml` — full demo stack (edge + simulator + cloud-api + Qdrant + Prometheus + Grafana + Tempo)

```bash
# Standalone cloud stack
CLOUD_GROQ_API_KEY=<key> docker compose -f docker-compose.cloud.yml up --build

# Full demo (edge + cloud)
CLOUD_GROQ_API_KEY=<key> docker compose -f docker-compose.demo.yaml up --build
```

---

## See Also

- `docs/agents_overview.md` — agent roles and responsibilities
- `docs/observability.md` — cloud metrics and traces
- `docs/OPERATIONS.md` — environment variable reference
- `src/cortexguard/cloud/graph/workflow.py` — LangGraph graph construction
- `src/cortexguard/cloud/graph/nodes.py` — individual node implementations
- `src/cortexguard/cloud/planner/factory.py` — LLM backend factory
- `src/cortexguard/cloud/retrieval/store.py` — RAG retrieval with outcome boosting
- `src/cortexguard/cloud/validation/plan_validator.py` — capability and confidence validation
