# Agents Overview

---

## Edge Agents (Reflex Layer)

These agents run locally on the edge device and are part of the current edge runtime.

### 1. SafetyAgent

- **Purpose**: Monitors the `Blackboard` (scene graph, anomaly events, state estimates) every orchestrator tick and evaluates hard-coded safety rules. Emits `SafetyCommand` with action `E-STOP | PAUSE | NOMINAL`.
- **Triggers immediate E-STOP on**: `HUMAN_PROXIMITY_VIOLATION`, `OVERHEAT_SMOKE_COMBO`, `OVERHEAT_SMOKE`
- **Where it lives**: `src/cortexguard/edge/safety_agent.py`

### 2. PolicyAgent

- **Purpose**: Generates `RemediationPolicy` objects in response to active `AnomalyEvent`s. Handles known anomalies via rules-based dispatch (e.g. overheat warnings) and delegates unknown/complex anomalies to the injected LLM policy engine (`MistralPolicyEngine`). Wraps output into high-priority `REMEDIATION` plans submitted to the `Orchestrator`.
- **Where it lives**: `src/cortexguard/edge/policy/policy_agent.py`

### 3. MaydayAgent

- **Purpose**: Edge-side escalation agent. When local recovery fails, builds a `MaydayPacket` from `Blackboard` context and sends it to the cloud via the injected `BaseCloudAgentClient`. Includes bounded timeouts, retry/backoff, and structured trace emission for observability.
- **Where it lives**: `src/cortexguard/edge/mayday_agent.py`

---

## Cloud Agents (Deliberative Layer)

The cloud tier receives escalations from the edge `MaydayAgent` and runs a LangGraph 5-node deliberative planning workflow.

### 1. Cloud Planner (Implemented)

- **Purpose**: Generate multi-step recovery plans for complex anomalies that edge local recovery could not resolve.
- **Example**: *"Repeated misgrasp on joint 2 — approach angle adjusted, retry grip with reduced torque, then resume task."*
- **How it works**: A LangGraph workflow runs five sequential nodes — persist incident → retrieve similar incidents (RAG over Qdrant) → generate candidate plan (LLM) → validate plan (capability + confidence checks) → route decision. Returns `plan_ready`, `needs_human`, or `no_safe_plan`.
- **Where it lives**: `src/cortexguard/cloud/`

See `docs/cloud_architecture.md` for a full breakdown of the LangGraph workflow and RAG pipeline.

---

## Future Cloud Agents

These agents are planned but not yet implemented.

### Explanation Agent (XAI Layer)

- **Purpose**: Translate low-level anomaly events into human-readable explanations for operators and debugging.
- **Example**: *"The torque sensor spiked during the stirring step — likely due to excessive resistance in the mixture. Recommend slowing the rotation speed."*

### Human-in-the-Loop Agent

- **Purpose**: Interface between the operator and the system. Routes high-uncertainty plans to operator approval before the edge executes them.
- **Example**:
  - *User: "Why did device 5 pause during task 2?"*
  - *LLM: "It detected an abnormal torque pattern during mixing, suggesting the whisk got stuck."*
