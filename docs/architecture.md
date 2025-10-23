


## Architecture summary

Edge performs low-latency sensing + many lightweight detectors → a fusion/score layer computes a unified risk score → control arbiter enforces safety (stop/slow) and dispatches to agents (automated recovery or human), while cloud services do heavier correlation, logging and continuous learning.



1. — High-level components

    * **Sensors (Edge)**: joint encoders, F/T, IMU, camera(s), depth, tactile, mic (audio), temperature, smoke/gas, weight/scale, magnetic/pressure where available.
    * **Intent / Context**: current BT node, expected subtask, expected durations, required resources.
    * **Local detectors (many)**: per-modality detectors (vision classifier, vision continuity, torque anomaly, audio contact detector, temp threshold), and step classifiers (is step done?). Use River/ONNX/quantized models to run on edge.
    * **Fusion engine (edge)**: subscribe to detectors; compute per-detector normalized scores, combine into a time-aware unified risk score and produce per-hypothesis evidence (provenance).
    * **Control arbiter / Safety layer (edge)**: enforces hard invariants (E-stop) and soft actions (slow, pause, park).
    * **Agent layer (cloud + edge)**: receives fused event → chooses recovery plan (rule-based, LLM reasoning assist, or RL policy), executes or asks human.
    * **Ingestion & logging (cloud)**: ingest events, store buffered video, telemetry, episodes for retraining.
    * **Model management (cloud/AWS)**: periodic training, push weights to edge.



2. — Fusion & ensemble design principles

    1. **Modality-specific detectors** — train or use detectors specialized for each sensor stream. Example detectors:
        - Vision: step-classifier CNN, object pose estimator, occlusion detector, visual drift detector (embedding distance).
        - Torque/F/T: windowed reconstruction AE or streaming Mahalanobis distance.
        - IMU / encoders: sudden velocity/accel spikes, no-motion detection.
        - Audio: contact/clatter classifier (mic picks up drop).
        - Thermal/smoke: threshold + trend detector.
        - Logs/intent: mismatch detector (e.g., expected pose vs observed).
    2. **Normalize outputs** — map each detector’s raw output to a standardized score in [0,1] where 0 = normal, 1 = highly anomalous. Also produce categorical tags (e.g., drop, collision, occlusion, smoke).
    3. **Temporal smoothing** — apply EMA to each score and compute short-window and long-window features (instant spike vs persistent drift).
    4. **Correlation matrix** — keep a time window matrix of detectors firing to learn common co-occurrences (e.g., torque spike + visual occlusion + audio thud => high-confidence drop).
    5. **Ensemble fusion**: combine via weighted sum + learned logistic meta-model (small classifier that consumes per-detector normalized scores, co-occurrence features, current intent, and outputs: risk_score ∈ [0,1] + likely_cause and confidence). The meta-model can be trained offline on labeled episodes (supervised) and used as a calibrator for unsupervised detectors.
    6. **Explainability** — always include top-3 contributing detectors and their raw evidence with the fused event.


3. — Decision logic: mapping risk → action

    Define three action tiers and safety invariants:

    * **Tier 0 (Immediate Safety — hard constraints)**
        - Conditions: catastrophic sensor thresholds (force > hardware limit, smoke sensor above emergency threshold, joint limit exceed).
        - Action: E-STOP (hardware e-stop if available), cut power to actuators, trigger alarm. No ML decision in the loop — deterministic.
    * **Tier 1 (High risk — automatic edge response)**
        - Conditions: risk_score ≥ R_high (e.g., 0.85) OR pattern-matched high-confidence dangerous cause like knife_drop_near_human, fire_detected.
        - Action: Stop motion immediately, set brakes, stream last N seconds of buffered video/telemetry to cloud, notify agents + humans with “urgent” tag.
    * **Tier 2 (Medium risk — adaptive action)**
        - Conditions: R_medium ≤ risk_score < R_high (e.g., 0.5–0.85), or multiple detectors weakly positive.
        - Action: Slow down / pause non-critical tasks (e.g., reduce arm speed by X%, lower torque limits), initiate targeted local recovery (e.g., re-grip attempt), or park non-involved arms. Also emit event to cloud for richer analysis.
    * **Tier 3 (Low risk / monitoring)**
        - Conditions: R_low ≤ risk_score < R_medium (e.g., 0.2–0.5)
        - Action: Increase vigilance: higher sampling, start buffering long video, notify human with low priority.
    * **Tier 4 TBD**
        - Conditions: Developing/ongoing situation identified at cloud level from larger context telemetry, but missed on edge
        - Action: Compose remediation plan on cloud and push locally with appropriate urgency level

    Set thresholds conservatively based on validation (start R_low=0.2, R_medium=0.5, R_high=0.85 as a baseline). The meta-model can also produce urgency and recommended_action.

    Important: the control arbiter must be the only component able to effect motion-level commands; agents request actions and the arbiter enforces safety invariants.


4. — Agents & handoffs

    * **Edge Safety Agent (reflex)**: on Tier 0/1 events, performs immediate actions locally (stop, retract, apply park pose). No cloud dependency.
    * **Edge Recovery Agent (fast)**: simple scripted recovery primitives (retry grip, re-align tool, small reposition) executed if time budget < cloud_latency_threshold.
    * **Cloud Decision Agent (deliberative)**: receives Tier2 events, uses retrieval (past episodes), small LLM + rules to propose a multi-step recovery plan (e.g., “move patty X mm, then re-stack, then resume”). Cloud agent returns structured plan; edge executes after vetting.
    * **Human-in-the-loop**: if plan risk > human_threshold or uncertainty high, send to operator UI with options: approve auto-execute, take manual control, or instruct different action.
    * **Learning Agent**: logs episodes and feedback (success/failure) used for retraining/fine tuning meta-models and classifiers.

    Agents communicate via defined APIs (HTTP/gRPC) and events on a message bus. All recovery actions are logged with provenance.



5. — Real-time constraints & latency budget

    * Hard control loop: < 50 ms (local low-level control). Must never depend on cloud.
    * Edge detection loop: 50–200 ms for lightweight detectors (River, ONNX models, quantized CNNs).
    * Fusion + decision (edge): 100–300 ms. This is where ensemble produces risk_score and arbiter acts.
    * Cloud-assisted decisions: acceptable at 500 ms — several seconds. But never used for immediate emergency response. Use cloud for richer multi-robot coordination or recovery that can wait.
    * Network planning: measure round-trip time (RTT) and set cloud_latency_threshold (e.g., 1.5s). If a decision must complete before deadline < cloud_latency_threshold, edge must proceed locally.



6. — Data model & event format (JSON)
    ```
    {
    "event_id": "uuid",
    "timestamp": "ISO8601",
    "robot_id": "kitchen-01",
    "bt_node": "Place_Top_Bun",
    "step_intent": "place_top_bun",
    "detector_scores": {
        "vision_step_classifier": 0.9,
        "vision_occlusion": 0.7,
        "ft_spike": 0.82,
        "audio_thud": 0.0,
        "smoke": 0.05
    },
    "fused": {
        "risk_score": 0.88,
        "likely_cause": "drop/topple",
        "confidence": 0.93,
        "top_evidence": [
        {"detector":"ft_spike", "score":0.82},
        {"detector":"vision_occlusion", "score":0.7}
        ]
    },
    "recommended_action": {"tier":1, "action":"STOP_AND_SAVE_BUFFER"}
    }
    ```



7. — Fusion / ensemble code sketch pseudocode

    This is a compact example showing detector normalization, fusion via logistic meta-model, and decision actions.

    ```
    # edge/fusion.py
    import time, math, uuid, json
    import numpy as np
    from collections import deque
    # placeholder classifier (trained offline, saved as small sklearn/logistic)
    try:
        import joblib
        META = joblib.load("meta_model.pkl")  # expects features in same order
    except Exception:
        META = None

    # detectors produce raw scores; define normalizers per detector type
    NORMALIZERS = {
        "vision_step": lambda x: float(np.clip(x,0,1)),
        "vision_occlusion": lambda x: float(min(1, x/0.8)),
        "ft_spike": lambda x: float(min(1, x/50)),  # if raw force spike in N
        "audio_thud": lambda x: float(np.clip(x,0,1)),
        "smoke": lambda x: float(np.clip(x/200, 0, 1)), # raw ppm -> normalized
    }

    # EMA state
    ema = {}
    alpha = 0.4

    def normalize_scores(raw_scores):
        out={}
        for k,v in raw_scores.items():
            norm = NORMALIZERS.get(k, lambda x: float(np.clip(x,0,1)))(v)
            # EMA smoothing
            prev = ema.get(k, norm)
            ema[k] = alpha*norm + (1-alpha)*prev
            out[k] = ema[k]
        return out

    def meta_predict(norm_scores, intent_features):
        # feature vector: ordered detectors + intent flags
        feat = []
        for k in sorted(norm_scores.keys()):
            feat.append(norm_scores[k])
        # add intent flags
        feat.extend([intent_features.get("is_place_bun",0)])
        X = np.array(feat).reshape(1,-1)
        if META:
            p = float(META.predict_proba(X)[0,1])
            cause = META.predict(X)[0]
            return p, cause
        else:
            # fallback: weighted sum heuristic
            weights = {k:0.2 for k in norm_scores}
            s = sum(weights[k]*norm_scores[k] for k in norm_scores)
            # bias with occlusion and ft_spike
            s += 0.3*norm_scores.get("ft_spike",0) + 0.2*norm_scores.get("vision_occlusion",0)
            return float(np.clip(s,0,1)), "heuristic_cause"

    def fuse_and_decide(raw_scores, intent):
        now = time.time()
        norm = normalize_scores(raw_scores)
        score, cause = meta_predict(norm, intent)
        # map score to tier
        tier = 3
        if score >= 0.85: tier = 1
        elif score >= 0.5: tier = 2
        elif score >= 0.2: tier = 3
        # recommended actions
        if tier == 1:
            action = "STOP_IMMEDIATE"
        elif tier == 2:
            action = "SLOW_AND_INSPECT"
        else:
            action = "MONITOR"
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": now,
            "robot_id": intent.get("robot_id"),
            "bt_node": intent.get("bt_node"),
            "detector_scores": norm,
            "fused": {
                "risk_score": score,
                "likely_cause": cause,
                "tier": tier,
                "recommended_action": action
            }
        }
        # log locally / push to message bus...
        with open("/tmp/anomaly_log.jsonl","a") as f:
            f.write(json.dumps(event)+"\n")
        return event
    ```



8. — Agent API sketch (edge ↔ cloud)

    Edge posts fused events to cloud, cloud replies with recommended structured plan. Always include action_id and require arbiter confirmation.

    Edge POST
    ```
    POST /api/anomaly
    body: { event_json }
    response: { plan_id, plan: [ {action, params, estimated_time, required_resources} ], confidence }
    ```

    Edge executes plan only after arbiter verifies resources & safety. If cloud fails to respond in ```cloud_latency_threshold```, edge executes fallback recovery primitives.



9. — Human-in-the-loop & UI

    * Provide real-time dashboard with video + highlighted evidence + recommended action (accept / modify / manual control).
    * For hazardous events (smoke/fire), send immediate push/phone alerts and log.
    * Provide “operator override” via secure API that arbiter respects.


10. — Testing & validation plan

    * **Unit tests**: detectors, normalizers, meta-model predictions.
    * **Integration tests**: replay labeled episodes (use NAB-like sensor streams + recorded video) to measure precision/recall for hazardous classes.
    * **Scenario tests (simulated)**: multi-task cooking with timed anomalies — measure how many items saved vs burned, false stop rate.
    * **Latency tests**: measure end-to-end time from sensor acquisition → decision → actuator command under different network conditions.
    * **Safety tests**: force hard-stop condition triggers, interlock validation.

    Metrics to report on README: detection latency, P/R for hazardous classes, false positive rate per hour, successful automatic recovery rate, MTTR.


11. — Deployment notes & model lifecycle

    * Run detectors as separate lightweight processes / containers on edge; orchestrator subscribes to their outputs.
    * Model updates: cloud trains meta-models/classifiers, signs artifacts, publishes to model registry, edge fetches with version check and atomic swap.
    * Telemetry retention: keep high-resolution buffers locally for last N seconds (circular buffer) and upload only around events to GCS/S3 to save bandwidth.
