# ---- Base Python image ----
FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/workspace/cortexguard/src" \
    LOG_LEVEL=INFO \
    LOG_JSON=true \
    UV_ENV=prod

WORKDIR /workspace/cortexguard

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN pip install uv && \
    uv sync --frozen --no-group dev --extra cloud --no-extra ml

ENV PATH="/workspace/cortexguard/.venv/bin:$PATH"

# Pre-download MiniLM embedder model so it survives container rebuilds
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

COPY src ./src

EXPOSE 8001

CMD ["uvicorn", "cortexguard.cloud.runtime:app", "--host", "0.0.0.0", "--port", "8001"]
