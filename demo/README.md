# CortexGuard Simulator Demo

## What the demo does
This demo runs a short simulation of the CortexGuard system using a small sample dataset.
It fuses sensor data (RGB images, depth images, and system logs) and streams it to the edge module.

**Key points:**
- Uses a small, committed sample dataset — no need to download the full dataset.
- Demonstrates the fusion pipeline and streaming engine.
- Streams fused records locally for inspection.

## How to run it
1. Activate the virtual environment. The path may vary depending on your setup:
   ```bash
   # Use the actual path to your virtual environment
    source .venv-local/bin/activate      # host machine
    source /opt/venvs/cortexguard/bin/activate  # container



2. Make the demo executable:
   ```bash
    chmod +x demo/run_demo.sh


3. Run the demo:
   ```bash
    /demo/run_demo.sh


The script will:
1. Fuse the sample dataset into FusedRecord objects.
2. Stream the fused records to the edge module in real time.


See `docs/simulator.md` for usage of fuser and streamer tasks.

## What to expect

* The simulator outputs logs showing progress of streaming.
* Each fused record includes:
    - Timestamp
    - RGB and depth image paths
    - Force, torque, and position values

* No anomaly scenarios or drift detection are included yet.
* The demo lasts roughly 10 seconds and is designed to be lightweight.

**Note**: This is a bootstrap/demo feature for local testing and development. Full datasets and cloud integration will be handled in later features.
