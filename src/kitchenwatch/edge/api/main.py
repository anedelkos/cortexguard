from fastapi import FastAPI

from kitchenwatch.common.logging_config import setup_logging
from kitchenwatch.edge.api import health, ingestion

# must be called before FastAPI app init
setup_logging()

app = FastAPI(
    title="KitchenWatch Edge API",
    description="Edge ingestion endpoint for fused records from the simulator.",
)

app.include_router(health.router)
app.include_router(ingestion.router, prefix="/api/v1")
