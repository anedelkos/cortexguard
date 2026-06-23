"""SageMaker ProcessingJob entry point for the retraining pipeline.

Fetches the DB URL from AWS Secrets Manager at runtime (never passed
in plaintext). Writes training metrics JSON plus a SageMaker ``code/``
bundle inside ``model.tar.gz`` for the Model Registry.

The scikit-learn built-in container serves inference by loading
``code/inference.py`` from the model archive, no custom Docker image
needed.

Usage (as a SageMaker ProcessingJob):
    python -m cortexguard.cloud.retraining.run

Environment variables:
    DB_URL_SECRET_ARN:  ARN of the Secrets Manager secret containing the DB URL
    OUTPUT_DIR:         directory to write training artifact (set by SageMaker)
    FETCH_DAYS:         only fetch telemetry rows newer than this many days (default 7)
"""

from __future__ import annotations

import json
import logging
import os
import tarfile
from datetime import UTC, datetime, timedelta
from typing import Any

try:
    from training_script import train  # sibling file (SageMaker)
except ImportError:
    from cortexguard.cloud.retraining.training_script import train  # as module

# Inference code bundled into model.tar.gz so the scikit-learn container
# can serve the model without a custom Docker image.
_INFERENCE_PY = r'''"""Stub predictor bundled with the model artifact.

Loaded by SageMaker Inference Toolkit in the scikit-learn container.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def model_fn(model_dir: str) -> dict[str, Any]:
    metadata_path = os.path.join(model_dir, "training_result.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path) as f:
            metadata: dict[str, Any] = json.load(f)
    else:
        metadata = {"model_id": "step_classifier_v1", "version": 0}
    logger.info("model_fn loaded %s v%s", metadata.get("model_id"), metadata.get("version"))
    return metadata


def input_fn(request_body: str | bytes, content_type: str = "application/json") -> dict[str, Any]:
    print(f"DEBUG: input_fn called. Type of request_body: {type(request_body)}")
    try:
        if isinstance(request_body, bytes):
            print(f"DEBUG: body length: {len(request_body)}, first 100 bytes (hex): {request_body[:100].hex()}")
            print(f"DEBUG: body raw bytes: {request_body}")
            print(f"DEBUG: body decoded (ignore errors): {request_body.decode('utf-8', errors='ignore')}")
        else:
            print(f"DEBUG: body length: {len(request_body)}, first 100 chars: {request_body[:100]}")
            print(f"DEBUG: body raw: {request_body}")
    except Exception as e:
        print(f"DEBUG: error logging body: {e}")
    print(f"DEBUG: content_type: {content_type}")

    if content_type == "application/json":
        if isinstance(request_body, bytes):
            return json.loads(request_body.decode("utf-8"))
        return json.loads(request_body)
    raise ValueError(f"Unsupported content_type: {content_type}")


def predict_fn(data: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    return {
        "predicted_outcome": "completed",
        "confidence": 0.95,
        "model_id": model.get("model_id", "unknown"),
        "model_version": model.get("version", 0),
        "num_features": len(data.get("sensor_snapshot", {})),
    }


def output_fn(prediction: dict[str, Any], accept: str = "application/json") -> str:
    if accept == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported accept type: {accept}")
'''

logger = logging.getLogger(__name__)


def _write_model_artifact(
    output_dir: str,
    training_result: dict[str, Any],
    inference_source: str = _INFERENCE_PY,
) -> str:
    """Write a SageMaker-compatible model.tar.gz into *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "training_result.json")
    with open(metrics_path, "w") as f:
        json.dump(training_result, f)

    code_dir = os.path.join(output_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    inference_path = os.path.join(code_dir, "inference.py")
    with open(inference_path, "w") as f:
        f.write(inference_source)

    setup_path = os.path.join(code_dir, "setup.py")
    with open(setup_path, "w") as f:
        f.write(
            "from setuptools import setup\n"
            'setup(name="inference", version="1.0.0", py_modules=["inference"])\n'
        )

    tarball_path = os.path.join(output_dir, "model.tar.gz")
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(metrics_path, arcname="training_result.json")
        tar.add(inference_path, arcname="code/inference.py")
        tar.add(setup_path, arcname="code/setup.py")
    return tarball_path


def _fetch_db_url(secret_arn: str) -> str:
    import boto3

    client = boto3.client("secretsmanager")
    response: Any = client.get_secret_value(SecretId=secret_arn)
    return str(response["SecretString"])


def _parse_records(rows: list[Any]) -> list[dict[str, Any]]:
    """Convert raw asyncpg records to structured dicts with parsed JSON."""
    result = []
    for r in rows:
        sensor_snapshot = r.get("sensor_snapshot_json") or "{}"
        result.append(
            {
                "device_id": r["device_id"],
                "key": r["key"],
                "outcome": r["outcome"],
                "timestamp": r["timestamp"],
                "sensor_snapshot": json.loads(sensor_snapshot),
            }
        )
    return result


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    secret_arn = os.environ["DB_URL_SECRET_ARN"]
    db_url = _fetch_db_url(secret_arn)
    output_dir = os.environ.get("OUTPUT_DIR", "/opt/ml/processing/output")
    os.makedirs(output_dir, exist_ok=True)
    fetch_days = int(os.environ.get("FETCH_DAYS", "7"))

    import asyncio

    import asyncpg  # type: ignore[import-untyped]

    async def _run() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            cutoff = datetime.now(UTC) - timedelta(days=fetch_days)
            rows = await conn.fetch(
                "SELECT device_id, key, outcome, timestamp, sensor_snapshot_json "
                "FROM step_telemetry "
                "WHERE timestamp >= $1 "
                "ORDER BY timestamp ASC",
                cutoff.isoformat(),
            )
        finally:
            await conn.close()

        parsed = _parse_records(rows)
        result = train(parsed)
        tarball_path = _write_model_artifact(output_dir, result)

        logger.info(
            "Training complete: rows=%d, window=%dd, metrics=%s, artifact=%s",
            len(parsed),
            fetch_days,
            os.path.join(output_dir, "training_result.json"),
            tarball_path,
        )

    asyncio.run(_run())


if __name__ == "__main__":
    _main()
