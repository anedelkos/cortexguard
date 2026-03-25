## Simulator Tasks

CortexGuard provides convenience Taskfile entries for building and testing the simulator components.

### Fuser
- **Purpose:** Combines raw device data from csv into fused JSONL records.
- **Entry:** `task simulate:fuse`
- **Usage:** Used during data prep to generate input for simulation or analysis.
- **Script:** wraps `demo/fuse_trial.py`
- **Output:** `data/fused/*.jsonl` (per trial)

### Streamer
- **Purpose:** Streams fused records to a local or remote edge receiver.
- **Entry:** `task simulate:stream`
- **Usage:** Runs the simulation loop using fused data and manifest files.
- **Script:** wraps `demo/simulate_stream.py`

### Chaos Scenario Streaming

- **Purpose:** Inject named anomaly scenarios into a synthetic baseline stream and observe the edge system's response in real time.
- **Script:** `demo/chaos_stream.py`
- **Scenarios:** defined in `data/anomaly_scenarios.yaml`

**List available scenarios:**
```bash
PYTHONPATH=src uv run python demo/chaos_stream.py --list
```

**Run a scenario against a live edge service:**
```bash
# Start the edge first
task edge:run

# Then in another terminal
PYTHONPATH=src uv run python demo/chaos_stream.py --scenario S0.1 --endpoint http://localhost:8080/api/v1/ingest
```

**Or use the demo Docker stack (no local Python install needed):**
```bash
SCENARIO=S0.1 docker compose -f docker-compose.demo.yaml up --build
```

| Scenario | Title | Expected Outcome |
|---|---|---|
| `S0.0` | No anomaly | Normal operation |
| `S0.1` | Human in Safety Radius | E-STOP |
| `S0.2` | Overheat + Smoke Combo | E-STOP |
| `S1.1` | Repeated Misgrasp | Escalate to cloud |
| `S2.3` | Sensor Freeze | Recovery |
| `S4.1` | Compound Fault (slip + occlusion + comm lag) | Recovery or escalate |

**Arguments:**

| Arg | Default | Description |
|---|---|---|
| `--scenario` | — | Scenario ID (e.g. `S0.1`) |
| `--endpoint` | `http://localhost:8080/api/v1/ingest` | Edge ingest endpoint |
| `--rate` | `1.0` | Streaming rate multiplier |
| `--list` | — | Print all scenarios and exit |

> **Slim mode:** The demo Docker image does not install torch or transformers. All 6 scenarios work without them — vision objects are injected via the ChaosEngine sidecar, bypassing the vision embedder entirely.
