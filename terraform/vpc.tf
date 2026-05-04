# Use the default VPC when no explicit vpc_id is provided.
# For a real production deployment, replace with a dedicated VPC.

data "aws_vpc" "selected" {
  id      = var.vpc_id != "" ? var.vpc_id : null
  default = var.vpc_id == "" ? true : null
}

data "aws_subnets" "selected" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.selected.id]
  }

  # When explicit subnet IDs are supplied, filter to them.
  dynamic "filter" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      name   = "subnet-id"
      values = var.subnet_ids
    }
  }
}

# Security group for the ALB — accepts HTTP and HTTPS from anywhere.
resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-alb"
  description = "Allow HTTP and HTTPS inbound to ALB"
  vpc_id      = data.aws_vpc.selected.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Security group for cloud-api ECS tasks — accepts traffic only from the ALB.
resource "aws_security_group" "ecs_tasks" {
  name        = "${local.name_prefix}-ecs-tasks"
  description = "Allow inbound from ALB only"
  vpc_id      = data.aws_vpc.selected.id

  ingress {
    from_port       = 8001
    to_port         = 8001
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Security group for the SQS worker — no inbound needed (it only polls SQS outbound).
resource "aws_security_group" "worker" {
  name        = "${local.name_prefix}-worker"
  description = "SQS worker - egress only"
  vpc_id      = data.aws_vpc.selected.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
