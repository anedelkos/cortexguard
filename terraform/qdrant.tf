resource "aws_service_discovery_private_dns_namespace" "main" {
  name = "${local.name_prefix}.local"
  vpc  = data.aws_vpc.selected.id
}

resource "aws_service_discovery_service" "qdrant" {
  name = "qdrant"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id

    dns_records {
      ttl  = 10
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

resource "aws_security_group" "qdrant" {
  name        = "${local.name_prefix}-qdrant"
  description = "Allow Qdrant REST from ECS tasks"
  vpc_id      = data.aws_vpc.selected.id

  ingress {
    from_port       = 6333
    to_port         = 6334
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id, aws_security_group.worker.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "efs_qdrant" {
  name        = "${local.name_prefix}-efs-qdrant"
  description = "Allow NFS from Qdrant ECS tasks"
  vpc_id      = data.aws_vpc.selected.id

  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.qdrant.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_efs_file_system" "qdrant" {
  encrypted = true

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }
}

resource "aws_efs_mount_target" "qdrant" {
  for_each = toset(data.aws_subnets.selected.ids)

  file_system_id  = aws_efs_file_system.qdrant.id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs_qdrant.id]
}


resource "aws_ecs_task_definition" "qdrant" {
  family                   = "${local.name_prefix}-qdrant"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  volume {
    name = "qdrant-storage"

    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.qdrant.id
      transit_encryption = "ENABLED"
    }
  }

  container_definitions = jsonencode([
    {
      name      = "qdrant"
      image     = "qdrant/qdrant:${var.qdrant_image_tag}"
      essential = true

      portMappings = [
        { containerPort = 6333, protocol = "tcp" },
        { containerPort = 6334, protocol = "tcp" },
      ]

      mountPoints = [
        {
          sourceVolume  = "qdrant-storage"
          containerPath = "/qdrant/storage"
          readOnly      = false
        }
      ]

      healthCheck = {
        command     = ["CMD-SHELL", "curl -sf http://localhost:6333/ || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.qdrant.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "qdrant"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "qdrant" {
  name             = "${local.name_prefix}-qdrant"
  cluster          = aws_ecs_cluster.main.id
  task_definition  = aws_ecs_task_definition.qdrant.arn
  desired_count    = 1
  launch_type      = "FARGATE"
  platform_version = "1.4.0"

  # Qdrant uses RocksDB which allows only one writer at a time.
  # Stop the old task before starting a new one to avoid LOCK file conflicts on EFS.
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  network_configuration {
    subnets          = data.aws_subnets.selected.ids
    security_groups  = [aws_security_group.qdrant.id]
    assign_public_ip = var.assign_public_ip
  }

  service_registries {
    registry_arn = aws_service_discovery_service.qdrant.arn
  }
}

locals {
  qdrant_url = "http://qdrant.${aws_service_discovery_private_dns_namespace.main.name}:6333"
}
