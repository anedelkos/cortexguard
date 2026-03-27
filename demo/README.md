# demo/

This directory contains scripts and assets for running and exploring CortexGuard scenarios.

| File | Purpose |
|------|---------|
| `chaos_stream.py` | Stream anomaly scenarios to a running edge service |
| `simulate_stream.py` | Lower-level stream replay utility |

Sample data lives in `data/` at the repo root. Scenario definitions are in `data/anomaly_scenarios.yaml`.

---

## Docker demo (recommended)

See the [Quick Demo](../README.md#-quick-demo) section in the root README. One command starts the full stack (edge service, simulator, Prometheus, Grafana, Tempo):

```bash
docker compose -f docker-compose.demo.yaml up --build
```

---

## chaos_stream.py

Streams a named anomaly scenario directly to a running edge service. Useful for testing outside Docker or targeting a specific scenario.

**List available scenarios:**
```bash
PYTHONPATH=src uv run python demo/chaos_stream.py --list
```

**Run a scenario:**
```bash
PYTHONPATH=src uv run python demo/chaos_stream.py --scenario S0.1
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--scenario` | `S0.1` | Scenario ID to stream |
| `--endpoint` | `http://localhost:8080/api/v1/ingest` | Edge ingest URL |
| `--rate` | `1.0` | Records per second |
| `--repeat` | `1` | Times to repeat; `0` = infinite |
| `--scenarios-file` | `data/anomaly_scenarios.yaml` | Path to scenario definitions |
| `--list` | — | Print all available scenarios and exit |

**Example — stream S1.1 twice at 2 req/s against a local edge:**
```bash
PYTHONPATH=src uv run python demo/chaos_stream.py \
  --scenario S1.1 \
  --endpoint http://localhost:8080/api/v1/ingest \
  --rate 2.0 \
  --repeat 2
```

Start the edge service first with `task edge:run`.
