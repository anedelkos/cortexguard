# VPC endpoints enabling SageMaker processing jobs in VPC mode to access
# S3, ECR, Secrets Manager, and CloudWatch Logs without a NAT gateway.
#
# These are required because when a processing job specifies VpcConfig,
# SageMaker places an ENI in the VPC subnets and all traffic goes through
# the VPC, no internet access without endpoints or a NAT gateway.

data "aws_route_table" "main" {
  vpc_id = data.aws_vpc.selected.id
  filter {
    name   = "association.main"
    values = ["true"]
  }
}

# Security group for interface VPC endpoints, allows HTTPS from the
# SageMaker processing job and existing ECS tasks.
resource "aws_security_group" "vpc_endpoints" {
  name        = "${local.name_prefix}-vpc-endpoints"
  description = "Allow HTTPS from SageMaker and ECS tasks to VPC endpoints"
  vpc_id      = data.aws_vpc.selected.id

  ingress {
    from_port = 443
    to_port   = 443
    protocol  = "tcp"
    security_groups = [
      aws_security_group.sagemaker.id,
      aws_security_group.ecs_tasks.id,
      aws_security_group.worker.id,
    ]
  }
}

# S3 Gateway Endpoint (free, no hourly charge)
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = data.aws_vpc.selected.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [data.aws_route_table.main.id]
}

# ECR API Endpoint
resource "aws_vpc_endpoint" "ecr_api" {
  vpc_id              = data.aws_vpc.selected.id
  service_name        = "com.amazonaws.${var.aws_region}.ecr.api"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = data.aws_subnets.selected.ids
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true
}

# ECR Docker Registry Endpoint
resource "aws_vpc_endpoint" "ecr_dkr" {
  vpc_id              = data.aws_vpc.selected.id
  service_name        = "com.amazonaws.${var.aws_region}.ecr.dkr"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = data.aws_subnets.selected.ids
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true
}

# Secrets Manager Endpoint
resource "aws_vpc_endpoint" "secrets_manager" {
  vpc_id              = data.aws_vpc.selected.id
  service_name        = "com.amazonaws.${var.aws_region}.secretsmanager"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = data.aws_subnets.selected.ids
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true
}

# CloudWatch Logs Endpoint (needed for SageMaker job logs)
resource "aws_vpc_endpoint" "logs" {
  vpc_id              = data.aws_vpc.selected.id
  service_name        = "com.amazonaws.${var.aws_region}.logs"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = data.aws_subnets.selected.ids
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true
}
