

# Design decisions overview

1. Edge vs Cloud
    * Edge: immediate, safety-critical decisions (fast rules + online learners)
    * Cloud: complex, multimodal reasoning; latency-tolerant
    * Trade-off: latency vs compute vs cost

2. Hybrid Ensemble
    * Fast rules: immediate, interpretable, deterministic
    * Lightweight online learners: adaptive, low-compute, detect drift
    * Heavy episodic models: multimodal, high-capacity reasoning in cloud
    * Why: balance speed, adaptability, and accuracy

3. Docker + Optional Kubernetes
    * Docker ensures reproducibility, containerized ML & agents
    * Kubernetes optional for scaling sensor data processing
    * Decision: GitHub demo uses Docker Compose; optional K8s for reliability in fleet-wide anomaly detection

4. Single-Cloud Choice (AWS)
    * AWS for both training and deliberative cloud inference: Python CDK/boto3 for dynamic provisioning, SageMaker for model lifecycle
    * Rationale: multi-cloud adds operational complexity without sufficient benefit at this scale; consolidating on AWS simplifies IAM, networking, and deployment

5. Explainability & Observability
    * XAI endpoint shows top modalities, anomaly timeline, LLM explanations
    * Prometheus/Grafana + ELK/OpenTelemetry for monitoring
    * Reasoning: Human operators and troubleshooters care about trustworthy, observable AI

6. Agent Coordination
    * Agents coordinated via CrewAI + vector DB
    * Multi-agent reasoning: SafetyAgent, PolicyAgent, MaydayAgent, StateEstimator, Orchestrator, HumanInterface
    * Why: Adds intelligent cutting-edge multi-agent reasoning and planning

7. Dataset Simulation & Testing
    * Public IoT dataset used to simulate real-time events
    * Artificial anomalies injected with ground truth
    * Enables reproducible evaluation & demo

8. Schema
    * AgentMessage, RecoveryPlan, AnomalyEvent
    * Standardization enforcement with Pydantic
    * Why: Interoperability and maintainability





# Key design choices regarding anomaly detection


1. Apply a single EMA per sensor signal (α = 0.1)

    🔍 What it means
    * EMA = Exponential Moving Average — a weighted moving average that gives more importance to recent samples.
    * A single EMA with `alpha=0.1` (heavy smoothing) is applied per sensor key across windows.

    💡 Why this is useful
    When we’re streaming sensor data (force, torque, temperature, vision embeddings, etc.), values fluctuate constantly — some spikes are real anomalies, others just noise.
    The low alpha suppresses transient noise while tracking the signal’s sustained trend.

    ⚙️ How it works
    ```
    # EMA Formula: new_ema = α * observation + (1-α) * old_ema
    # α = 0.1 → heavy smoothing, slow response to transient spikes
    ema_state[key] = alpha * observation + (1 - alpha) * ema_state[key]
    ```

    ✅ Result: fewer false positives from transient noise; sustained anomalies accumulate in the smoothed value.


2. Optimistic fallback / Partial connection: Edge acts immediately, cloud later reconciles

    🔍 What it means
    This is a network resilience pattern — how your edge device behaves when connectivity to AWS cloud is slow or temporarily lost.
        * Optimistic fallback = The edge assumes it can make the right decision locally if the cloud isn’t responsive.
        * Partial connection = The system continues operating in “degraded mode,” caching events and syncing later.

    💡 Why it matters
    You can’t rely on cloud roundtrips for real-time reactions. Even 200–500 ms delay might ruin a task or cause safety risks.

    So instead:
    1. Edge makes the immediate safety or recovery decision locally.
    2. It logs the decision and event.
    3. When the cloud reconnects, it sends logs for reconciliation — cloud replays what happened and updates its models or policies.

    ⚙️ Example
    A processing step fails and the cloud link lags.
    ```
    if network_latency > 0.5 or no_ack_from_cloud:
        execute_local_recovery("retry_step")
        cache_event("local_decision", timestamp, action="retry_step")
    else:
        ask_cloud_for_plan()
    ```

    Later:
    ```
    if connection_restored:
        sync_cached_events_to_cloud()
        reconcile_models()
    ```
    ✅ Benefits:
        - Safe even when offline
        - No data loss
        - Cloud eventually learns from the event
        So, Edge = Reflexes, Cloud = Memory + Learning.


3. ~~Use a Behavior Tree for runtime control, but allow LLM or RL agents to modify, expand, or repair BTs dynamically~~

    > **Superseded.** The Behavior Tree approach was replaced by a `Plan` / `PlanStep` execution model coordinated by the `Orchestrator`. Plans are priority-queued and preemptable; the `PolicyAgent` generates `RemediationPolicy` objects containing corrective steps which are wrapped into `Plan`s and executed via `StepExecutor`. This provides the same reactive and self-repairing properties without the complexity of runtime BT graph patching.






# 🧠 System Overview — Edge–Cloud Cooperative Architecture
1. Mission

    The system executes complex task plans broken into ordered steps received from a cloud orchestration system.
    Each task step (e.g., “process item”, “place component”) is executed, monitored, and validated locally on the edge device for correctness, timing, and safety.


2. Control Flow

    1. **Cloud → Edge**:
    The cloud issues a structured task plan, describing steps and constraints.
    2. **Edge Executor**:
    The `Orchestrator` schedules `Plan`s via an async priority queue. Each `Plan` contains ordered `PlanStep`s executed by `StepExecutor`, polling classifiers and sensors to verify completion.
    3. **Monitoring**:
    During execution, local anomaly detectors and state estimators continuously watch sensor streams (vision, force, torque, temperature, etc.).
    4. **Progress & Validation**:
    Step classifiers answer “is this step complete / failed?” questions.
    The next step only starts once completion is confirmed.


3. Safety & Reflex Layer (Edge)

    Edge maintains reflex-level reactions independent of cloud latency.

    * **Immediate safety interrupts**:
    Human intrusion, collision risk, or force overload triggers instant stop or slowdown (<100 ms reaction).
    * **Local retries**:
    If a step fails (e.g., object slipped), the `PolicyAgent` generates a `RemediationPolicy` with corrective steps (re-grip, reposition, etc.) which the `Orchestrator` executes as a high-priority plan.
    * **EMA filtering**:
    Each sensor signal is smoothed with a single EMA (α = 0.1) to distinguish transient noise from sustained anomalies.
    * **Resource and deadline awareness**:
    The local scheduler prioritizes urgent tasks using estimated time-to-done (TTD), time-to-failure (TTF), and safety margins.

4. Cloud Layer (Reasoning & Coordination)

    The cloud is deliberative, not time-critical. It:
    * Aggregates telemetry and anomaly logs from multiple edge devices.
    * Runs heavier models (LLMs, RL policies) to:
        - Diagnose complex failures.
        - Generate multi-step recovery plans.
        - Optimize task policies or adjust thresholds.
    * Updates the edge’s policies or parameters during low-urgency periods.
    * Maintains fleet-level analytics and retraining datasets.

    Cloud decisions are integrated asynchronously — the edge acts immediately, while the cloud reconciles or improves policies later (optimistic fallback).

5. Edge–Cloud Cooperation Rules
    Case	                                Decision Location	                    Reason
    Urgent safety / latency < 200 ms	    Edge only	                            Cloud too slow
    Simple retry / recovery	                Edge	                                Known local pattern
    Complex anomaly / unknown failure	    Cloud	                                Requires reasoning / planning
    Network degraded	                    Edge acts → Cloud reconciles later	    Optimistic fallback
    Long-term adaptation	                Cloud	                                Heavy models, fleet analysis


6. Anomaly Detection Layers
    Layer	Models	                                Timescale	            Purpose
    Edge	Isolation Forests, lightweight CNNs	    milliseconds–seconds	Reflex detection, physical anomalies
    Cloud	LLM reasoning, deep analytics	        seconds–minutes	        Pattern discovery, global optimization

    Each detector outputs normalized anomaly scores with short/long EMA smoothing.
    A **fusion module** correlates signals (vision + force + audio → event cause and confidence).

7. Fault Tolerance & Recovery

    1. **Failure → Event Bus**:
    If a step fails or anomaly is detected, the `Orchestrator` pauses the current plan and emits an event.
    2. **Policy Agent (local)**:
    Generates a `RemediationPolicy` with corrective steps (retry step, re-stack component) executed as a high-priority plan.
    3. **Escalation**:
    If retries fail or the anomaly is unknown, `MaydayAgent` escalates to the cloud.
    4. **Cloud Decision Agent**:
    Generates structured multi-step recovery plans and pushes them back to the edge.
    5. **Synchronization**:
    Edge executes immediately; cloud logs, learns, and may update future policies.


8. Plan Execution Dynamics
    * `Plan`/`PlanStep` model provides deterministic control, safety, and observability.
    * Anomaly detectors can interrupt active plans via the `Orchestrator`'s preemption mechanism.
    * Higher-priority `REMEDIATION` plans preempt lower-priority `TASK` plans.
    * Only one plan holds execution control at any time; the `Orchestrator` enforces this.

9. Design Principle

    * **Never block for cloud decisions.**
        Edge handles real-time safety and bounded-time recovery.
        Cloud handles heavy reasoning, coordination, and learning.
        If disconnected, the edge keeps operating safely using cached policies and local inference.


10. Key Advantages
    * **Low latency, high safety** — local reflexes for human and environmental risks.
    * **Scalable intelligence** — cloud learns from fleet data and updates edge logic.
    * **Resilient operation** — optimistic fallback ensures continued service under network loss.
    * **Adaptive planning** — `PolicyAgent` generates remediation plans; `MaydayAgent` escalates to cloud for complex scenarios.
    * **Explainable decisions** — every anomaly and action is logged with cause, context, and source detector.




# 💡 Design Takeaways
    Plans/Steps = “what should happen”
    Step Classifiers = “did it happen as planned?”
    Anomaly Detectors = “did anything weird happen?”
    Policy Agent = “how do we fix it safely?”
    Orchestrator = “who has control right now?”
    These work together like a nervous system:
        Reflex (local anomaly → pause/retry)
        Cerebellum (policy agent → stabilize)
        Cortex (cloud planner → update policy)


* Coding this for Jetson Orin:
    - Run anomaly detectors and classifiers as independent microservices (Docker containers) publishing events to MQTT.
    - Run the Orchestrator in a lightweight asyncio loop.
    - Let the orchestrator (FastAPI + asyncio) handle routing and state.



# High level folder structure
src/cortexguard/
├── edge/
│   ├── detectors/
│   ├── models/
│   ├── policy/
│   ├── api/
│   ├── observability/
│   ├── utils/
│   ├── safety_agent.py
│   ├── policy_agent.py
│   ├── mayday_agent.py
│   ├── orchestrator.py
│   ├── step_executor.py
│   └── runtime.py
├── simulation/
├── common/
└── core/
