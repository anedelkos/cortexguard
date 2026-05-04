#!/usr/bin/env bash
# Build, push, and deploy the cloud-api to ECS/Fargate.
#
# Prerequisites:
#   - aws CLI configured with sufficient IAM permissions
#   - docker logged in to ECR (this script handles that)
#   - terraform initialized in this directory (terraform init)
#
# Usage:
#   ./deploy.sh <aws-account-id> <region> [image-tag]
#
# Example:
#   ./deploy.sh 123456789012 eu-west-1 v1.0.0

set -euo pipefail

ACCOUNT_ID="${1:?Usage: deploy.sh <aws-account-id> <region> [image-tag]}"
REGION="${2:?Usage: deploy.sh <aws-account-id> <region> [image-tag]}"
IMAGE_TAG="${3:-$(git rev-parse --short HEAD)}"

ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
# ECR repo name must match the Terraform resource (project-environment-cloud-api)
PROJECT="${TF_VAR_project:-cortexguard}"
ENVIRONMENT="${TF_VAR_environment:-demo}"
ECR_REPO="${ECR_REGISTRY}/${PROJECT}-${ENVIRONMENT}-cloud-api"

echo "==> Authenticating Docker to ECR"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "==> Building cloud-api image"
docker build \
  --platform linux/amd64 \
  -f docker/cloud.Dockerfile \
  -t "${ECR_REPO}:${IMAGE_TAG}" \
  -t "${ECR_REPO}:latest" \
  .

echo "==> Pushing to ECR"
docker push "${ECR_REPO}:${IMAGE_TAG}"
docker push "${ECR_REPO}:latest"

echo "==> Running terraform apply"
terraform -chdir=terraform apply \
  -var "aws_region=${REGION}" \
  -var "image_tag=${IMAGE_TAG}" \
  -auto-approve

echo ""
echo "==> Deployment complete"
terraform -chdir=terraform output alb_dns_name
