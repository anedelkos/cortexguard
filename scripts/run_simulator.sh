#!/bin/bash
set -euo pipefail

# --- Default values ---
RATE=0.5
REPEAT=0
MANIFEST_PATH=""
ENDPOINT="http://edge:8080/api/v1/ingest"
VERBOSE=""

# --- Parse CLI arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --rate)
      RATE="$2"
      shift 2
      ;;
    --repeat)
      REPEAT="$2"
      shift 2
      ;;
    --endpoint)
      ENDPOINT="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE="$2"
      shift 2
      ;;
    *)
      echo "❌ Unknown argument: $1"
      echo "Usage: $0 [--manifest <path>] [--rate <float>] [--repeat <int>] [--endpoint <str>] [--verbose <str>]"
      exit 1
      ;;
  esac
done


# --- Detect environment ---
if [ -d "/workspace/kitchenwatch" ]; then
    ROOT_DIR="/workspace/kitchenwatch"
    echo "🧩 Running inside container"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    ROOT_DIR="$(dirname "$SCRIPT_DIR")"
    echo "💻 Running locally"
fi

# --- Accept optional argument ---
DEFAULT_MANIFEST="$ROOT_DIR/data/manifests/full_dataset_manifest.yaml"
MANIFEST_PATH="${MANIFEST_PATH:-$DEFAULT_MANIFEST}"

# --- Define paths ---
FUSED_DIR="$ROOT_DIR/data/fused"
DEMO_DIR="$ROOT_DIR/demo"

# --- Export for subprocesses ---
export MANIFEST_PATH
export FUSED_DIR
export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR/src"

echo "📂 Using root directory: $ROOT_DIR"
echo "📄 Manifest path: $MANIFEST_PATH"
echo "🎛️ Rate: $RATE"
echo "🔁 Repeat: $REPEAT"
echo "🔗 Endpoint: ${ENDPOINT:-<not set>}"
echo "📜 Verbose: ${VERBOSE:-<not set>}"

mkdir -p "$FUSED_DIR"

# --- Fuse trials if needed ---
echo "⚡ Fusing trials if needed..."
python3 "$DEMO_DIR/fuse_trial.py" \
        --manifest "$MANIFEST_PATH"


# --- Run simulator ---
echo "▶️ Starting simulator (default: infinite repeat)..."
python3 "$DEMO_DIR/simulate_stream.py" \
    --manifest "$MANIFEST_PATH" \
    --repeat "$REPEAT" \
    --rate "$RATE" \
    --endpoint "$ENDPOINT" \
    --verbose "$VERBOSE"
