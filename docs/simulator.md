## Simulator Tasks

CortexGuard provides convenience Taskfile entries for building and testing the simulator components.

### Fuser
- **Purpose:** Combines raw device data from csv into fused JSONL records.
- **Entry:** `task simulate:fuse`
- **Usage:** Used during data prep to generate input for simulation or analysis.
- - **Script:** wraps `demo/fuse_trial.py`
- **Output:** `data/fused/*.jsonl` (per trial)

### Streamer
- **Purpose:** Streams fused records to a local or remote edge receiver.
- **Entry:** `task simulate:stream`
- **Usage:** Runs the simulation loop using fused data and manifest files.
- **Script:** wraps `demo/simulate_stream.py`
