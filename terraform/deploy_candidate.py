"""Deploys a new model as the challenger variant at 10% traffic.

Triggered by EventBridge when a new model version is registered.
The new model becomes the "challenger" variant at 10% traffic. Champion stays
at 90% unchanged. A separate promote_challenger Lambda promotes the
challenger to champion after a monitoring window if metrics are healthy.

Records the previous champion in SSM so rollback can revert.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

sagemaker = boto3.client("sagemaker")
ssm = boto3.client("ssm")

MODEL_PACKAGE_GROUP = os.environ["MODEL_PACKAGE_GROUP"]
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
PREFIX = os.environ["PREFIX"]
MODEL_BUCKET = os.environ["MODEL_BUCKET"]
CONTAINER_IMAGE = os.environ["SAGEMAKER_SKLEARN_IMAGE"]
EXECUTION_ROLE_ARN = os.environ["EXECUTION_ROLE_ARN"]

CHAMPION_SSM = f"/{PREFIX}/champion/previous-version"
CANDIDATE_SSM = f"/{PREFIX}/candidate/pending"


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    new_package_arn: str | None = None
    detail_raw = event.get("detail")
    if isinstance(detail_raw, dict):
        new_package_arn = detail_raw.get("ModelPackageArn")
    if not new_package_arn:
        return {"status": "error", "message": "missing ModelPackageArn"}

    # Capture current champion before any changes. This is the rollback target
    current_champion = _get_current_champion()
    if current_champion:
        ssm.put_parameter(
            Name=CHAMPION_SSM,
            Value=json.dumps(current_champion),
            Type="String",
            Overwrite=True,
        )

    # Auto-approve the model package
    sagemaker.update_model_package(
        ModelPackageArn=new_package_arn,
        ModelApprovalStatus="Approved",
    )

    # Resolve the new model artifact
    new_version = _resolve_version(new_package_arn)
    model_data_url = _resolve_artifact(new_package_arn)

    # Deploy candidate model as challenger variant at 10%
    candidate_model_name = f"{PREFIX}-step-classifier-candidate-v{new_version}"
    _create_model(candidate_model_name, model_data_url)
    config_name = _create_endpoint_config_with_candidate(candidate_model_name, new_version)
    _update_endpoint(config_name)

    # Record pending candidate for promote_challenger
    ssm.put_parameter(
        Name=CANDIDATE_SSM,
        Value=json.dumps(
            {
                "version": new_version,
                "model_name": candidate_model_name,
                "deployed_at": datetime.now(UTC).isoformat(),
            }
        ),
        Type="String",
        Overwrite=True,
    )

    return {
        "status": "candidate_deployed",
        "candidate_version": new_version,
        "endpoint_config": config_name,
    }


def _get_current_champion() -> dict[str, object] | None:
    """Return version and ARN of the currently approved champion."""
    pkgs = sagemaker.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
        ModelApprovalStatus="Approved",
    )
    versions = pkgs.get("ModelPackageSummaryList", [])
    if not versions:
        return None
    champ = versions[0]
    return {
        "version": champ["ModelPackageVersion"],
        "arn": champ["ModelPackageArn"],
    }


def _resolve_version(new_package_arn: str) -> int:
    desc = sagemaker.describe_model_package(ModelPackageName=new_package_arn)
    return int(desc.get("ModelPackageVersion", 0))


def _resolve_artifact(package_arn: str) -> str:
    desc = sagemaker.describe_model_package(ModelPackageName=package_arn)
    containers = desc.get("InferenceSpecification", {}).get("Containers", [])
    if containers:
        url: str | None = containers[0].get("ModelDataUrl")
        return url or ""
    logger.warning(
        "No containers found in model package %s, falling back to bootstrap artifact",
        package_arn,
    )
    return f"s3://{MODEL_BUCKET}/artifacts/model.tar.gz"


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


def _resolve_data_capture_config(desc: dict[str, Any]) -> dict[str, Any]:
    dcc = desc.get("DataCaptureConfig")
    if isinstance(dcc, dict):
        return dcc
    logger.warning("Current endpoint config has no DataCaptureConfig. Enabling with 100%% sampling")
    return {
        "EnableCapture": True,
        "InitialSamplingPercentage": 100,
        "DestinationS3Uri": f"s3://{MODEL_BUCKET}/data-capture",
        "CaptureOptions": [
            {"CaptureMode": "Input"},
            {"CaptureMode": "Output"},
        ],
    }


def _create_endpoint_config_with_candidate(candidate_model_name: str, version: int) -> str:
    """Create an endpoint config with champion at 90% and candidate (challenger) at 10%."""
    desc = sagemaker.describe_endpoint_config(
        EndpointConfigName=sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)[
            "EndpointConfigName"
        ]
    )
    variants = []
    for pv in desc.get("ProductionVariants", []):
        if pv["VariantName"] == "challenger":
            pv["ModelName"] = candidate_model_name
            pv["InitialVariantWeight"] = 10
        elif pv["VariantName"] == "champion":
            pv["InitialVariantWeight"] = 90
        pv.pop("VariantStatus", None)
        variants.append(pv)

    config_name = f"{PREFIX}-step-clf-config-candidate-v{version}"
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
