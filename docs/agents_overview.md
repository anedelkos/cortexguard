# Agents Overview

---

## Edge Agents (Implemented)

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

## Cloud Agents (Planned — Deliberative Layer)

> **Not yet implemented.** The following agents are planned for the deliberative cloud layer. The edge `MaydayAgent` already handles escalation to the cloud; these agents would live on the AWS side and respond to escalations.

### 1. Recovery Planner (Reasoning Agent)

- **Purpose**: Generate multi-step recovery plans for complex anomalies where edge rule-based logic isn't sufficient.
- **Example**: *"Sensor malfunction detected in joint 2. Retry calibration, then resume task from step 3."*
- **How it works**: LLM is given context (anomaly type, system state, recent task history) and generates a structured recovery plan. The plan is pushed back to the edge for execution.

### 2. Explanation Agent (XAI Layer)

- **Purpose**: Translate low-level anomaly events into human-readable explanations for operators and debugging.
- **Example**: *"The torque sensor spiked during the stirring step — likely due to excessive resistance in the mixture. Recommend slowing the rotation speed."*
- **How it works**: LLM receives structured anomaly event JSON and produces a plain-language summary with a recommended action.

### 3. Human-in-the-Loop Agent

- **Purpose**: Interface between the operator and the system. Allows operators to query system state in natural language and receive dynamic explanations.
- **Example**:
  - *User: "Why did device 5 pause during task 2?"*
  - *LLM: "It detected an abnormal torque pattern during mixing, suggesting the whisk got stuck."*

### 4. Multi-Agent Reasoning Coordinator

- **Purpose**: Coordinates multiple cloud sub-agents (data integrity, recovery, communication) to handle complex failure scenarios that require parallel reasoning.
- **Example prompt**: *"Decide which agent should handle this anomaly event: safety-critical, mechanical, or data drift."*
