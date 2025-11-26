# ---- Base Python image ----
FROM python:3.12-slim AS base

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/workspace/kitchenwatch/src" \
    LOG_LEVEL=INFO \
    LOG_JSON=true \
    UV_ENV=prod

WORKDIR /workspace/kitchenwatch

# System deps
RUN apt-get update &&  \
    apt-get install -y --no-install-recommends \
    bash curl build-essential &&  \
    rm -rf /var/lib/apt/lists/*

# Copy only dependency metadata first (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (using uv if available)
RUN pip install uv && \
    uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install --system -r requirements.txt && \
    rm requirements.txt


# Copy source
COPY src ./src

# Expose FastAPI port
EXPOSE 8080

# Run the FastAPI app
CMD ["uvicorn", "kitchenwatch.edge.runtime:get_api_app", "--host", "0.0.0.0", "--port", "8080"]
