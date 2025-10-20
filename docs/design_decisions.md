

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
    * Kubernetes optional for scaling agents, orchestrator, cloud detector
    * Decision: GitHub demo uses Docker Compose; optional K8s for reliability in fleet-wide anomaly detection

4. Multi-Cloud Choice
    * AWS for training: Python CDK/boto3 for dynamic provisioning
    * GCP for detection: Terraform for declarative, stable deployment

5. Explainability & Observability
    * XAI endpoint shows top modalities, anomaly timeline, LLM explanations
    * Prometheus/Grafana + ELK/OpenTelemetry for monitoring
    * Reasoning: Human operators and troubleshooters care about trustworthy, observable AI

6. Agent Coordination
    * Agents coordinated via CrewAI + vector DB
    * Multi-agent reasoning: SafetyAgent, RecoveryAgent, StateEstimator, Orchestrator, HumanInterface
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


1. Apply EMA short (1s) and long (30s) per detector

    🔍 What it means
    * EMA = Exponential Moving Average — a weighted moving average that gives more importance to recent samples.
    * We're applying two EMAs per detector:
        - Short-term EMA (1s window) → reacts quickly to changes (fast reflexes)
        - Long-term EMA (30s window) → tracks stable baseline over time

    💡 Why this is useful
    When we’re streaming sensor data (force, torque, temperature, vision embeddings, etc.), values fluctuate constantly — some spikes are real anomalies, others just noise.
    The dual-EMA trick helps separate sustained anomalies from transient blips.

    ⚙️ How it works
    Example with a force sensor:
    ```
    ema_short = EMA(window=1.0)  # 1 second
    ema_long = EMA(window=30.0)  # 30 seconds

    for t, value in sensor_stream:
        short = ema_short.update(value)
        long = ema_long.update(value)
        delta = abs(short - long)
        if delta > threshold:
            trigger_anomaly("force_drift", magnitude=delta)
    ```

    - If the short EMA rises sharply above the long EMA → sudden force spike (likely anomaly)
    - If the short EMA slowly drifts → trend change (wear, sensor drift)
    - If both move together → normal dynamics

    Do this on every detector (force, temperature, optical flow, etc.), giving you both instant reactions and trend awareness.
    ✅ Result: fewer false positives, faster reaction to real problems.


2. Optimistic fallback / Partial connection: Edge acts immediately, cloud later reconciles

    🔍 What it means
    This is a network resilience pattern — how your edge device (the robot) behaves when connectivity to the cloud (AWS/GCP) is slow or temporarily lost.
        * Optimistic fallback = The edge assumes it can make the right decision locally if the cloud isn’t responsive.
        * Partial connection = The system continues operating in “degraded mode,” caching events and syncing later.

    💡 Why it matters
    You can’t rely on cloud roundtrips for real-time reactions. Even 200–500 ms delay might ruin a task or cause safety risks.

    So instead:
    1. Edge makes the immediate safety or recovery decision locally.
    2. It logs the decision and event.
    3. When the cloud reconnects, it sends logs for reconciliation — cloud replays what happened and updates its models or policies.

    ⚙️ Example
    Patty starts burning and cloud link lags.
    ```
    if network_latency > 0.5 or no_ack_from_cloud:
        execute_local_recovery("flip_patty_again")
        cache_event("local_decision", timestamp, action="retry_flip")
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


3. Use a Behavior Tree for runtime control, but allow LLM or RL agents to modify, expand, or repair BTs dynamically
    🔍 What it means

    You’re using a Behavior Tree (BT) to run recipes — a graph of tasks like:
    ```
    CookBurger
    ├── PickBun
    ├── FlipPatty
    ├── StackTomato
    └── ApplySauce
    ```

    But what happens when an unseen anomaly occurs?
    Example: “Sauce bottle missing” — not something you explicitly encoded in your BT.

    Here’s where the meta-agent (LLM or RL policy) comes in:
    It can edit or extend the BT at runtime to handle new situations.

    ⚙️ Example
    1. BT running node ApplySauce
    2. Vision detects “no sauce bottle”
    3. BT node fails → sends failure_event to orchestrator
    4. LLM meta-agent analyzes event context and proposes patch:
    ```
    {
    "add_subtree_after": "ApplySauce",
    "new_subtree": {
        "name": "HandleMissingSauce",
        "children": [
            {"action": "search_inventory"},
            {"condition": "if_found"},
            {"action": "refill_dispenser"},
            {"action": "retry_apply_sauce"}
            ]
        }
    }
    ```
    5. Orchestrator patches BT graph in memory:
    ```bt_executor.insert_subtree("ApplySauce", new_subtree)```
    6. Execution continues seamlessly.

    💡 Why this is powerful
    - Keeps execution reactive (BT)
    - Enables self-repairing logic
    - Lets AI agents generalize to new failure modes without manual reprogramming
    - RL version can learn which patches succeed more often

🧠 Meta-Agent Behavior
    Role	            What it edits	                        Input	                                    Output
    LLM Meta-Agent	    Adds or repairs BT subtrees	            event context, logs, ontology of actions	BT patch
    RL Policy Updater	Tunes retry thresholds, priorities	    telemetry, rewards	                        new BT parameters
    Human Supervisor	Approves or refines proposed patches	proposed diff	                            accepted patch

    ✅ Result: a living Behavior Tree — evolving and learning safely, bounded by rules (no arbitrary code execution).






# 🧠 System Overview — Edge–Cloud Cooperative Architecture
1. Mission

    The system executes complex cooking or assembly recipes broken into ordered steps received from a cloud ordering system.
    Each recipe step (e.g., “flip patty”, “apply sauce”) is executed, monitored, and validated locally on the robot (edge device) for correctness, timing, and safety.


2. Control Flow

    1. **Cloud → Edge**:
    The cloud issues a structured recipe plan, describing tasks and constraints.
    2. **Edge Executor**:
    A Behavior Tree (BT) orchestrates local step execution, polling classifiers and sensors to verify completion.
    Each node corresponds to a discrete action (pick, flip, stack, apply).
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
    If a step fails (e.g., object slipped), the recovery manager attempts predefined retry routines (re-grip, reposition, etc.).
    * **EMA filtering**:
    Each detector applies short (1 s) and long (30 s) Exponential Moving Averages to distinguish transient spikes from sustained anomalies.
    * **Resource and deadline awareness**:
    The local scheduler prioritizes urgent tasks using estimated time-to-done (TTD), time-to-failure (TTF), and safety margins.

4. Cloud Layer (Reasoning & Coordination)

    The cloud is deliberative, not time-critical. It:
    * Aggregates telemetry and anomaly logs from multiple edge devices.
    * Runs heavier models (LLMs, RL policies) to:
        - Diagnose complex failures.
        - Generate multi-step recovery plans.
        - Optimize task policies or adjust thresholds.
    * Updates the edge’s behavior trees or parameters during low-urgency periods.
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
    If a step fails or anomaly is detected, the BT pauses and emits an event.
    2. **Recovery Agent (local)**:
    Executes known recovery primitives (retry flip, re-stack bun).
    3. **Escalation**:
    If retries fail or anomaly is unknown, event escalates to the cloud.
    4. **LLM Meta-Agent (cloud)**:
    Generates structured recovery plans or patches Behavior Trees dynamically (e.g., insert “refill sauce” step).
    5. **Synchronization**:
    Edge executes immediately; cloud logs, learns, and may update future BT templates.


8. Behavior Tree Dynamics
    * BT provides deterministic control, safety, and observability.
    * Anomaly detectors can interrupt active BT nodes via event signals.
    * A meta-agent (LLM/RL) can edit or extend BTs at runtime to handle new failure modes safely.
    * Only one controller (BT or recovery agent) holds motion control at any time.

9. Design Principle

    * **Never block for cloud decisions.**
        Edge handles real-time safety and bounded-time recovery.
        Cloud handles heavy reasoning, coordination, and learning.
        If disconnected, the edge keeps operating safely using cached policies and local inference.


10. Key Advantages
    * **Low latency, high safety** — local reflexes for human and environmental risks.
    * **Scalable intelligence** — cloud learns from fleet data and updates edge logic.
    * **Resilient operation** — optimistic fallback ensures continued service under network loss.
    * **Adaptive planning** — BTs evolve through LLM/RL meta-agents for new scenarios.
    * **Explainable decisions** — every anomaly and action is logged with cause, context, and source detector.




# 💡 Design Takeaways
    Behavior Trees = “what should happen”
    Step Classifiers = “did it happen as planned?”
    Anomaly Detectors = “did anything weird happen?”
    Recovery Agent = “how do we fix it safely?”
    Orchestrator = “who has control right now?”
    These work together like a nervous system:
        Reflex (local anomaly → pause/retry)
        Cerebellum (recovery agent → stabilize)
        Cortex (cloud planner → update policy)


* Coding this for Jetson Orin:
    - Run anomaly detectors and classifiers as independent microservices (Docker containers) publishing events to MQTT.
    - Run the Behavior Tree executor in a lightweight Python loop.
    - Let the orchestrator (FastAPI + asyncio) handle routing and state.



# High level folder structure
src/
├── edge/
│   ├── behavior_tree/
│   ├── detectors/
│   ├── safety_agent.py
│   ├── recovery_agent.py
│   └── orchestrator.py
├── cloud/
│   ├── reasoning_agent.py
│   ├── analytics/
│   ├── retraining/
│   └── llm_planner.py
└── shared/
    ├── schemas/
    ├── utils/
    └── event_bus.py
