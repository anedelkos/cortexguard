# Agents Overview

---

## Edge Agents (Reflex Layer)

These agents run locally on the edge device and are part of the current edge runtime.

### 1. SafetyAgent

- **Purpose**: Monitors the `Blackboard` (scene graph, anomaly events, state estimates) every orchestrator tick and evaluates hard-coded safety rules. Emits `SafetyCommand` with action `E-STOP | PAUSE | NOMINAL`.
- **Triggers immediate E-STOP on**: `HUMAN_PROXIMITY_VIOLATION`, `OVERHEAT_SMOKE_COMBO`, `OVERHEAT_SMOKE`
- **Where it lives**: `src/cortexguard/edge/safety_agent.py`

### 2. PolicyAgent

- **Purpose**: Generates `RemediationPolicy` objects in response to active `AnomalyEvent`s. Handles known anomalies via rules-based dispatch (e.g. overheat warnings) and delegates unknown/complex anomalies to the injected LLM policy engine (`LLMPolicyEngine`). Wraps output into high-priority `REMEDIATION` plans submitted to the `Orchestrator`.
- **Where it lives**: `src/cortexguard/edge/policy/policy_agent.py`

### 3. MaydayAgent

- **Purpose**: Edge-side escalation agent. When local recovery fails, builds a `MaydayPacket` from `Blackboard` context and sends it to the cloud via the injected `BaseCloudAgentClient`. Includes bounded timeouts, retry/backoff, and structured trace emission for observability.
- **Where it lives**: `src/cortexguard/edge/mayday_agent.py`

---

## Cloud Agents (Deliberative Layer)

The cloud tier receives escalations from the edge `MaydayAgent` and runs a LangGraph 5-node deliberative planning workflow.

### 1. Cloud Planner (Implemented)

- **Purpose**: Generate multi-step recovery plans for complex anomalies that edge local recovery could not resolve.
- **Example**: *"Repeated actuation failure on axis 2: approach angle adjusted, retry with reduced force, then resume task."*
- **How it works**: A LangGraph workflow runs five sequential nodes, persist incident → retrieve similar incidents (RAG over Qdrant) → generate candidate plan (LLM) → validate plan (capability + confidence checks) → route decision. Returns `plan_ready`, `needs_human`, or `no_safe_plan`.
- **Where it lives**: `src/cortexguard/cloud/`

See `docs/cloud_architecture.md` for a full breakdown of the LangGraph workflow and RAG pipeline.

---

## MCP Server (Operator Interface)

- **Purpose**: Gives an operator or AI assistant a structured inspection and control interface over the cloud planner, without touching the operational edge-to-cloud path.
- **Where it lives**: `src/cortexguard/cloud/mcp_server.py`
- **Connect via**: `claude mcp add cortexguard -- docker exec -i cortexguard-cloud-api python -m cortexguard.cloud.mcp_server`

### Resources (read-only)

| Resource | Description |
|---|---|
| `capability_registry` | Full capability catalog from `common/capability_registry.yaml` |
| `recent_incident_summaries` | Last N planning episodes with decision and outcome |
| `latest_planner_decision` | Most recent planning result in full detail |

### Tools (actions)

| Tool | Description |
|---|---|
| `get_latest_incident` | Fetch the most recent planning incident, decision, plan, rationale, and retrieved similar incidents with similarity scores. Use this first when an alert fires. |
| `create_remediation_plan` | Run the full LangGraph planning workflow for a given anomaly context |
| `validate_plan` | Run the validation layer against a provided Plan: returns pass/fail and errors |
| `explain_plan` | LLM-generated plain-English explanation of a Plan's steps and rationale |
| `propose_alternative_plan` | Rerun planning with a constraint to avoid a prior approach |
| `record_operator_resolution` | Capture what the operator did and whether it worked; re-embeds in Qdrant for future RAG retrieval |

### Operator workflow for `needs_human`

When the cloud planner returns `needs_human`, the operator connects via Claude Code and asks questions naturally. Claude calls the MCP tools automatically:

1. *"A needs_human alert fired, use get_latest_incident to see what happened"* → calls `get_latest_incident`, returns full incident detail including RAG-retrieved similar incidents with similarity scores
2. *"Explain the plan it was going to run"* → calls `explain_plan`
3. *"Try without the recalibration step"* → calls `propose_alternative_plan`
4. After resolving manually → calls `record_operator_resolution` to feed the outcome back into the RAG store
