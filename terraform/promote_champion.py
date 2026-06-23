"""Auto-approves new model packages and promotes champion.

Triggered by EventBridge when a new model version is registered.
Creates a versioned model + EndpointConfig, then updates the endpoint.
Records the previous champion in SSM so the rollback Lambda can revert.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

sagemaker = boto3.client("sagemaker")
ssm = boto3.client("ssm")

MODEL_PACKAGE_GROUP = os.environ["MODEL_PACKAGE_GROUP"]
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
PREFIX = os.environ["PREFIX"]
MODEL_BUCKET = os.environ["MODEL_BUCKET"]
CONTAINER_IMAGE = os.environ["SAGEMAKER_SKLEARN_IMAGE"]
EXECUTION_ROLE_ARN = os.environ["EXECUTION_ROLE_ARN"]

SSM_PARAM = f"/{PREFIX}/champion/previous-version"


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    new_package_arn: str | None = event.get("detail", {}).get("ModelPackageArn")  # type: ignore[call-overload]
    if not new_package_arn:
        return {"status": "error", "message": "missing ModelPackageArn"}

    # Capture the current champion before promoting — this is what rollback
    # will restore.  Query Model Registry (pre-approval) so the list returns
    # the champion about to be replaced, not the package we are promoting.
    current_champion = _get_current_champion()
    if current_champion:
        ssm.put_parameter(
            Name=SSM_PARAM,
            Value=json.dumps(current_champion),
            Type="String",
            Overwrite=True,
        )

    # Auto-approve
    sagemaker.update_model_package(
        ModelPackageArn=new_package_arn,
        ModelApprovalStatus="Approved",
    )

    # Get details of the new approved model
    pkgs = sagemaker.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=2,
        ModelApprovalStatus="Approved",
    )
    versions = pkgs.get("ModelPackageSummaryList", [])
    if not versions:
        return {"status": "no_models"}

    new_champion = versions[0]
    new_version = new_champion["ModelPackageVersion"]
    model_data_url = _resolve_artifact(new_champion["ModelPackageArn"])
    model_name = _model_name("champion", new_version)

    _create_model(model_name, model_data_url)
    config_name = _create_endpoint_config(model_name, new_version)
    _update_endpoint(config_name)

    return {
        "status": "promoted",
        "champion_version": new_version,
        "endpoint_config": config_name,
    }


def _model_name(variant: str, version: int) -> str:
    return f"{PREFIX}-step-classifier-{variant}-v{version}"


def _get_current_champion() -> dict[str, object] | None:
    """Return version and ARN of the currently approved champion.

    Queried before the new package is approved, so the result is the
    champion about to be replaced (not the package we just approved).
    """
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
    # Describe the current config to replicate its production variants
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

    config_name = f"{PREFIX}-step-clf-config-v{version}"
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
