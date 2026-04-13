# ---- Base Python image ----
FROM python:3.12-slim-bookworm AS base

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/workspace/cortexguard/src" \
    LOG_LEVEL=INFO \
    LOG_JSON=true \
    UV_ENV=prod

WORKDIR /workspace/cortexguard

# System deps
RUN apt-get update &&  \
    apt-get install -y --no-install-recommends \
    bash curl build-essential &&  \
    rm -rf /var/lib/apt/lists/*

# Copy only dependency metadata first (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies via uv sync.
# Pass --build-arg SLIM=true to skip heavy ML packages (torch/transformers).
ARG SLIM=false
RUN pip install uv && \
    if [ "$SLIM" = "true" ]; then \
      uv sync --frozen --no-group dev --no-extra ml; \
    else \
      uv sync --frozen --no-group dev --extra ml; \
    fi

ENV PATH="/workspace/cortexguard/.venv/bin:$PATH"


# Copy source
COPY src ./src

# Expose FastAPI port
EXPOSE 8080

# Run the FastAPI app
CMD ["uvicorn", "cortexguard.edge.runtime:get_api_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
