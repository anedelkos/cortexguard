resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

locals {
  sqs_queue_url = aws_sqs_queue.mayday.url

  common_environment = [
    { name = "CLOUD_LLM_BACKEND",         value = var.llm_backend },
    { name = "CLOUD_EMBEDDER_BACKEND",     value = "miniLM" },
    { name = "CLOUD_VECTOR_STORE_BACKEND", value = "qdrant" },
    { name = "CLOUD_QDRANT_URL",           value = local.qdrant_url },
    { name = "CLOUD_INCIDENT_STORE",       value = "postgres" },
    { name = "CLOUD_SQS_QUEUE_URL",        value = local.sqs_queue_url },
    { name = "CLOUD_SQS_REGION",           value = var.aws_region },
    { name = "LOG_JSON",                   value = "true" },
    { name = "LOG_LEVEL",                  value = "INFO" },
  ]

  common_secrets = [
    {
      name      = "CLOUD_GROQ_API_KEY"
      valueFrom = aws_secretsmanager_secret.groq_api_key.arn
    },
    {
      name      = "CLOUD_ANTHROPIC_API_KEY"
      valueFrom = aws_secretsmanager_secret.anthropic_api_key.arn
    },
    {
      name      = "CLOUD_API_KEY"
      valueFrom = aws_secretsmanager_secret.cloud_api_key.arn
    },
    {
      name      = "CLOUD_DB_URL"
      valueFrom = aws_secretsmanager_secret.db_url.arn
    },
  ]

  api_log_config = {
    logDriver = "awslogs"
    options = {
      "awslogs-group"         = aws_cloudwatch_log_group.cloud_api.name
      "awslogs-region"        = var.aws_region
      "awslogs-stream-prefix" = "cloud-api"
    }
  }

  worker_log_config = {
    logDriver = "awslogs"
    options = {
      "awslogs-group"         = aws_cloudwatch_log_group.worker.name
      "awslogs-region"        = var.aws_region
      "awslogs-stream-prefix" = "worker"
    }
  }
}

# ── Cloud API task ────────────────────────────────────────────────────────────

resource "aws_ecs_task_definition" "cloud_api" {
  family                   = "${local.name_prefix}-cloud-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.task_cpu)
  memory                   = tostring(var.task_memory)
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "cloud-api"
      image     = local.ecr_image
      essential = true

      portMappings = [{ containerPort = 8001, protocol = "tcp" }]

      environment = local.common_environment
      secrets     = local.common_secrets

      logConfiguration = local.api_log_config

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8001/healthz/live || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

resource "aws_ecs_service" "cloud_api" {
  name            = "${local.name_prefix}-cloud-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.cloud_api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.selected.ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = var.assign_public_ip
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.cloud_api.arn
    container_name   = "cloud-api"
    container_port   = 8001
  }

  depends_on = [aws_lb_listener.http]
}

# ── Planning worker task ──────────────────────────────────────────────────────

resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.name_prefix}-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.task_cpu)
  memory                   = tostring(var.task_memory)
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "worker"
      image     = local.ecr_image
      essential = true

      command = ["python", "-m", "cortexguard.cloud.worker"]

      environment = local.common_environment
      secrets     = local.common_secrets

      logConfiguration = local.worker_log_config

      healthCheck = {
        command     = ["CMD-SHELL", "find /tmp/worker-heartbeat -mmin -2 || exit 1"]
        interval    = 60
        timeout     = 5
        retries     = 3
        startPeriod = 120
      }
    }
  ])
}

resource "aws_ecs_service" "worker" {
  name            = "${local.name_prefix}-worker"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = var.worker_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.selected.ids
    security_groups  = [aws_security_group.worker.id]
    assign_public_ip = var.assign_public_ip
  }
}
