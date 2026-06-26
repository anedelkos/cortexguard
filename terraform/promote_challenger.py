"""Promotes the challenger candidate to champion after a monitoring window.

Triggered by a scheduled EventBridge rule (every 15 minutes).
Checks CloudWatch metrics for the challenger variant. If healthy and the
monitoring window has elapsed, promotes the challenger to champion at 90%.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

sagemaker = boto3.client("sagemaker")
cloudwatch = boto3.client("cloudwatch")
ssm = boto3.client("ssm")

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
PREFIX = os.environ["PREFIX"]
MODEL_BUCKET = os.environ["MODEL_BUCKET"]
CONTAINER_IMAGE = os.environ["SAGEMAKER_SKLEARN_IMAGE"]
EXECUTION_ROLE_ARN = os.environ["EXECUTION_ROLE_ARN"]

CANDIDATE_SSM = f"/{PREFIX}/candidate/pending"
MONITORING_WINDOW_MINUTES = int(os.environ.get("CANDIDATE_MONITORING_WINDOW_MINUTES", "30"))
ERROR_RATE_THRESHOLD = int(os.environ.get("CANDIDATE_ERROR_RATE_THRESHOLD", "5"))
LATENCY_MS_THRESHOLD = int(os.environ.get("CANDIDATE_LATENCY_MS_THRESHOLD", "2000"))
MIN_CHALLENGER_INVOCATIONS = int(os.environ.get("CANDIDATE_MIN_INVOCATIONS", "1"))


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    try:
        param = ssm.get_parameter(Name=CANDIDATE_SSM)
        candidate = json.loads(param["Parameter"]["Value"])
    except ClientError:
        return {"status": "no_candidate_pending"}

    candidate_version: int = candidate["version"]
    candidate_model_name: str = candidate["model_name"]
    deployed_at = datetime.fromisoformat(candidate["deployed_at"])

    elapsed = datetime.now(UTC) - deployed_at
    if elapsed < timedelta(minutes=MONITORING_WINDOW_MINUTES):
        remaining = MONITORING_WINDOW_MINUTES - int(elapsed.total_seconds() / 60)
        return {"status": "monitoring_window_not_met", "remaining_minutes": remaining}

    healthy, metrics_report = _check_challenger_metrics()
    if not healthy:
        logger.warning("Candidate v%s failed health check: %s", candidate_version, metrics_report)
        return {
            "status": "health_check_failed",
            "candidate_version": candidate_version,
            "reason": metrics_report,
        }

    champion_model_name = f"{PREFIX}-step-classifier-champion-v{candidate_version}"
    _create_model(champion_model_name, candidate_model_name)
    config_name = _create_endpoint_config_with_champion(
        champion_model_name, candidate_model_name, candidate_version
    )
    _update_endpoint(config_name)

    try:
        ssm.delete_parameter(Name=CANDIDATE_SSM)
    except ClientError:
        logger.warning("Failed to delete SSM candidate state (will be overwritten by next deploy)")

    logger.info("Promoted candidate v%s to champion, config=%s", candidate_version, config_name)
    return {
        "status": "promoted",
        "champion_version": candidate_version,
        "endpoint_config": config_name,
    }


def _check_challenger_metrics() -> tuple[bool, str]:
    now = datetime.now(UTC)
    start = now - timedelta(minutes=MONITORING_WINDOW_MINUTES)

    invocations_resp = cloudwatch.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName="InvocationCount",
        Dimensions=[
            {"Name": "EndpointName", "Value": ENDPOINT_NAME},
            {"Name": "VariantName", "Value": "challenger"},
        ],
        StartTime=start,
        EndTime=now,
        Period=300,
        Statistics=["Sum"],
    )
    invocations = sum(dp["Sum"] for dp in invocations_resp.get("Datapoints", []))
    if invocations < MIN_CHALLENGER_INVOCATIONS:
        return (
            False,
            f"Challenger invocations: {invocations} (minimum: {MIN_CHALLENGER_INVOCATIONS})",
        )

    error_resp = cloudwatch.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName="ModelInvocation4XXErrors",
        Dimensions=[
            {"Name": "EndpointName", "Value": ENDPOINT_NAME},
            {"Name": "VariantName", "Value": "challenger"},
        ],
        StartTime=start,
        EndTime=now,
        Period=300,
        Statistics=["Sum"],
    )
    error_sum = sum(dp["Sum"] for dp in error_resp.get("Datapoints", []))
    if error_sum > ERROR_RATE_THRESHOLD:
        return False, f"4XX errors: {error_sum} (threshold: {ERROR_RATE_THRESHOLD})"

    error5_resp = cloudwatch.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName="ModelInvocation5XXErrors",
        Dimensions=[
            {"Name": "EndpointName", "Value": ENDPOINT_NAME},
            {"Name": "VariantName", "Value": "challenger"},
        ],
        StartTime=start,
        EndTime=now,
        Period=300,
        Statistics=["Sum"],
    )
    error5_sum = sum(dp["Sum"] for dp in error5_resp.get("Datapoints", []))
    if error5_sum > 0:
        return False, f"5XX errors: {error5_sum}"

    latency_resp = cloudwatch.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName="ModelLatency",
        Dimensions=[
            {"Name": "EndpointName", "Value": ENDPOINT_NAME},
            {"Name": "VariantName", "Value": "challenger"},
        ],
        StartTime=start,
        EndTime=now,
        Period=300,
        Statistics=["p99"],
    )
    p99_values = [dp["p99"] for dp in latency_resp.get("Datapoints", []) if "p99" in dp]
    if p99_values and max(p99_values) > LATENCY_MS_THRESHOLD:
        return False, f"p99 latency: {max(p99_values):.0f}ms (threshold: {LATENCY_MS_THRESHOLD}ms)"

    return (
        True,
        f"ok (invocations={invocations}, errors={error_sum}, p99={max(p99_values) if p99_values else 'N/A'}ms)",
    )


def _create_model(model_name: str, source_model_name: str) -> None:
    try:
        desc = sagemaker.describe_model(ModelName=source_model_name)
    except ClientError as e:
        logger.error("Source model %s not found: %s", source_model_name, e)
        raise

    container = desc["PrimaryContainer"]
    try:
        sagemaker.delete_model(ModelName=model_name)
    except ClientError:
        pass
    sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": container["Image"],
            "ModelDataUrl": container["ModelDataUrl"],
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            },
        },
        ExecutionRoleArn=EXECUTION_ROLE_ARN,
    )


def _resolve_data_capture_config(desc: dict[str, Any]) -> dict[str, Any]:
    dcc = desc.get("DataCaptureConfig")
    if isinstance(dcc, dict):
        return dcc
    return {
        "EnableCapture": True,
        "InitialSamplingPercentage": 100,
        "DestinationS3Uri": f"s3://{MODEL_BUCKET}/data-capture",
        "CaptureOptions": [
            {"CaptureMode": "Input"},
            {"CaptureMode": "Output"},
        ],
    }


def _create_endpoint_config_with_champion(
    champion_model_name: str, challenger_model_name: str, candidate_version: int
) -> str:
    desc = sagemaker.describe_endpoint_config(
        EndpointConfigName=sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)[
            "EndpointConfigName"
        ]
    )
    variants = []
    for pv in desc.get("ProductionVariants", []):
        if pv["VariantName"] == "champion":
            pv["ModelName"] = champion_model_name
            pv["InitialVariantWeight"] = 90
        elif pv["VariantName"] == "challenger":
            pv["ModelName"] = challenger_model_name
            pv["InitialVariantWeight"] = 10
        pv.pop("VariantStatus", None)
        variants.append(pv)

    config_name = f"{PREFIX}-step-clf-config-champion-v{candidate_version}-{int(time.time())}"
    sagemaker.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=variants,
        DataCaptureConfig=_resolve_data_capture_config(desc),
    )
    return config_name


def _update_endpoint(config_name: str) -> None:
    sagemaker.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=config_name,
    )
