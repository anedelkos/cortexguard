# --- S3 bucket for model artifacts ---

resource "aws_s3_bucket" "models" {
  bucket        = "${local.name_prefix}-models"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    id     = "expire-old-artifacts"
    status = "Enabled"
    filter {
      prefix = ""
    }
    expiration {
      days = 90
    }
  }
}

# Upload retraining source code to S3 so the pipeline can reference it.
resource "aws_s3_object" "retraining_run" {
  bucket = aws_s3_bucket.models.id
  key    = "code/run.py"
  source = "${path.module}/../src/cortexguard/cloud/retraining/run.py"
  etag   = filemd5("${path.module}/../src/cortexguard/cloud/retraining/run.py")
}

resource "aws_s3_object" "retraining_training_script" {
  bucket = aws_s3_bucket.models.id
  key    = "code/training_script.py"
  source = "${path.module}/../src/cortexguard/cloud/retraining/training_script.py"
  etag   = filemd5("${path.module}/../src/cortexguard/cloud/retraining/training_script.py")
}

# --- SageMaker execution role ---

data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name               = "${local.name_prefix}-sagemaker-execution"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json
}

data "aws_iam_policy_document" "sagemaker_execution" {
  # S3 read/write for model artifacts and source code
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.models.arn}/*"]
  }
  # Model Registry write
  statement {
    actions = [
      "sagemaker:AddTags",
      "sagemaker:CreateModelPackage",
      "sagemaker:DescribeModelPackage",
    ]
    resources = ["*"]
  }
  # Secrets Manager: fetch DB URL at runtime
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.db_url.arn]
  }
  # VPC resources for SageMaker network
  statement {
    actions = [
      "ec2:CreateNetworkInterface",
      "ec2:DescribeNetworkInterfaces",
      "ec2:DeleteNetworkInterface",
    ]
    resources = ["*"]
  }
  # ECR cross-account pull for SageMaker built-in images
  statement {
    actions = [
      "ecr:BatchGetImage",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = ["arn:aws:ecr:${var.aws_region}:${data.aws_sagemaker_prebuilt_ecr_image.sklearn.registry_id}:repository/sagemaker-scikit-learn"]
  }
}

resource "aws_iam_policy" "sagemaker_execution" {
  name   = "${local.name_prefix}-sagemaker-execution"
  policy = data.aws_iam_policy_document.sagemaker_execution.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = aws_iam_policy.sagemaker_execution.arn
}

# --- SageMaker VPC Security Group ---
# Allows outbound to RDS (5432) and general egress for VPC endpoints.

resource "aws_security_group" "sagemaker" {
  name        = "${local.name_prefix}-sagemaker"
  description = "Allow SageMaker processing job to reach RDS and VPC endpoints"
  vpc_id      = data.aws_vpc.selected.id

  egress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.selected.cidr_block]
  }

  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.selected.cidr_block]
  }
}

# --- Model Package Group ---

resource "aws_sagemaker_model_package_group" "models" {
  model_package_group_name = "${local.name_prefix}-edge-models"
  tags = {
    Name        = "${local.name_prefix}-edge-models"
    Service     = "sagemaker"
    Environment = var.environment
  }
}

# --- SageMaker Pipeline ---

resource "aws_sagemaker_pipeline" "retraining" {
  pipeline_name         = "${local.name_prefix}-retraining"
  pipeline_display_name = "cortexguard-step-classifier-retraining"
  role_arn              = aws_iam_role.sagemaker_execution.arn

  pipeline_definition = templatefile("${path.module}/pipeline_definition.json.tpl", {
    sagemaker_sklearn_image       = local.sagemaker_sklearn_image
    sagemaker_model_monitor_image = local.sagemaker_model_monitor_image
    sagemaker_role_arn       = aws_iam_role.sagemaker_execution.arn
    code_bucket              = aws_s3_bucket.models.bucket
    output_bucket            = aws_s3_bucket.models.bucket
    secret_arn               = aws_secretsmanager_secret.db_url.arn
    model_package_group_name = aws_sagemaker_model_package_group.models.model_package_group_name
    subnet_ids               = jsonencode(data.aws_subnets.selected.ids)
    sagemaker_sg_ids         = jsonencode([aws_security_group.sagemaker.id])
  })

  depends_on = [
    aws_sagemaker_model_package_group.models,
    aws_security_group.sagemaker,
  ]
}

# --- Weekly schedule via EventBridge Scheduler ---

data "aws_iam_policy_document" "scheduler_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scheduler_execution" {
  name               = "${local.name_prefix}-scheduler-retraining"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume_role.json
}

data "aws_iam_policy_document" "scheduler_execution" {
  statement {
    actions   = ["sagemaker:StartPipelineExecution"]
    resources = [aws_sagemaker_pipeline.retraining.arn]
  }
}

resource "aws_iam_policy" "scheduler_execution" {
  name   = "${local.name_prefix}-scheduler-retraining"
  policy = data.aws_iam_policy_document.scheduler_execution.json
}

resource "aws_iam_role_policy_attachment" "scheduler_execution" {
  role       = aws_iam_role.scheduler_execution.name
  policy_arn = aws_iam_policy.scheduler_execution.arn
}

resource "aws_scheduler_schedule" "retraining" {
  name = "${local.name_prefix}-retraining-weekly"
  flexible_time_window {
    mode = "OFF"
  }
  schedule_expression = "cron(0 6 ? * 1 *)"  # Monday 06:00 UTC

  target {
    arn      = aws_sagemaker_pipeline.retraining.arn
    role_arn = aws_iam_role.scheduler_execution.arn

    sagemaker_pipeline_parameters {
      pipeline_parameter {
        name  = "FetchDays"
        value = "7"
      }
    }
  }
}
