# ---- Base Python image ----
FROM python:3.12-slim-bookworm


# ---- Environment setup ----
ENV PATH="/opt/venvs/kitchenwatch/bin:$PATH" \
    VENV_PATH=/opt/venvs/kitchenwatch \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/workspace/kitchenwatch/src" \
    MANIFEST_PATH="/workspace/kitchenwatch/data/manifests/sample_dataset_manifest.yaml" \
    FUSED_DIR="/workspace/kitchenwatch/data/fused"

# ---- Set working directory ----
WORKDIR /workspace/kitchenwatch

# ---- System dependencies ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash curl wget build-essential git && \
    rm -rf /var/lib/apt/lists/*

# ---- Copy dependency manifests ----
COPY pyproject.toml uv.lock ./

# ---- Create venv and install deps via uv ----
RUN python -m venv $VENV_PATH && \
    $VENV_PATH/bin/pip install --upgrade pip && \
    $VENV_PATH/bin/pip install "uv>=0.9.5" && \
    $VENV_PATH/bin/python -m uv sync --frozen --active

# ---- Copy project files ----
COPY src ./src
COPY demo ./demo
COPY data ./data
COPY scripts ./scripts

# ---- Ensure run script is executable ----
RUN chmod +x ./scripts/run_simulator.sh

# ---- Default entrypoint ----
ENTRYPOINT ["./scripts/run_simulator.sh"]
CMD []
