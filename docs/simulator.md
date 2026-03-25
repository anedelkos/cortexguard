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
| `--scenario` | `S0.1` | Scenario ID (e.g. `S0.2`) |
| `--endpoint` | `http://localhost:8080/api/v1/ingest` | Edge ingest endpoint |
| `--rate` | `1.0` | Records per second |
| `--repeat` | `1` | Times to repeat the stream (`0` = infinite loop) |
| `--list` | — | Print all scenarios and exit |

> **Slim mode:** The demo Docker image does not install torch or transformers. All 6 scenarios work without them — vision detections and occlusion metadata are serialised directly into the `WindowedFusedRecord` payload and processed by the edge without requiring the vision embedder.

> **Docker demo loop:** The Docker demo stack runs with `--repeat 0` by default so Grafana receives a continuous stream of data.
