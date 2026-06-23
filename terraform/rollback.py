"""Rolls back to the previous champion model package.

Triggered by SNS (drift alarm or endpoint error alarm).
Reads the previous champion from SSM, creates a versioned model and
EndpointConfig, then updates the endpoint.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

sagemaker = boto3.client("sagemaker")
ssm = boto3.client("ssm")

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
PREFIX = os.environ["PREFIX"]
MODEL_PACKAGE_GROUP = os.environ["MODEL_PACKAGE_GROUP"]
CONTAINER_IMAGE = os.environ["SAGEMAKER_SKLEARN_IMAGE"]
EXECUTION_ROLE_ARN = os.environ["EXECUTION_ROLE_ARN"]
MODEL_BUCKET = os.environ["MODEL_BUCKET"]

SSM_PARAM = f"/{PREFIX}/champion/previous-version"


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    alarm_name = "unknown"
    try:
        records: list[dict[str, Any]] = event.get("Records", [{}])  # type: ignore[assignment]
        record = records[0]
        sns_msg: dict[str, Any] = json.loads(str(record.get("Sns", {}).get("Message", "{}")))  # type: ignore[call-overload]
        alarm_name = str(sns_msg.get("detail", {}).get("alarmName", "unknown"))  # type: ignore[call-overload]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass
    print(f"Rollback triggered by: {alarm_name}")

    try:
        param = ssm.get_parameter(Name=SSM_PARAM)
        prev = json.loads(param["Parameter"]["Value"])
    except ClientError:
        print("No previous champion recorded, cannot roll back")
        return {"status": "no_rollback_target"}

    version = prev["version"]
    model_data_url = _resolve_artifact(prev["arn"])
    model_name = f"{PREFIX}-step-classifier-champion-v{version}"

    _create_model(model_name, model_data_url)
    config_name = _create_endpoint_config(model_name, version)
    _update_endpoint(config_name)

    print(f"Rolled back to champion version {version}, config {config_name}")
    return {
        "status": "rolled_back",
        "previous_version": version,
        "alarm": alarm_name,
        "endpoint_config": config_name,
    }


def _create_model(model_name: str, model_data_url: str) -> None:
    try:
        sagemaker.delete_model(ModelName=model_name)
    except ClientError:
        pass
    sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": CONTAINER_IMAGE,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            },
        },
        ExecutionRoleArn=EXECUTION_ROLE_ARN,
    )


def _create_endpoint_config(model_name: str, version: int) -> str:
    desc = sagemaker.describe_endpoint_config(
        EndpointConfigName=sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)[
            "EndpointConfigName"
        ]
    )
    variants = []
    for pv in desc.get("ProductionVariants", []):
        if pv["VariantName"] == "champion":
            pv["ModelName"] = model_name
        pv.pop("VariantStatus", None)
        variants.append(pv)

    config_name = f"{PREFIX}-step-clf-config-v{version}-rollback"
    sagemaker.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=variants,
        DataCaptureConfig=desc.get("DataCaptureConfig", {}),
    )
    return config_name


def _update_endpoint(config_name: str) -> None:
    sagemaker.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=config_name,
    )


def _resolve_artifact(package_arn: str) -> str:
    desc = sagemaker.describe_model_package(ModelPackageName=package_arn)
    containers = desc.get("InferenceSpecification", {}).get("Containers", [])
    if containers:
        url: str | None = containers[0].get("ModelDataUrl")
        return url or ""
    return f"s3://{MODEL_BUCKET}/artifacts/model.tar.gz"
