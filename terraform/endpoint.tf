# ── SageMaker endpoint, Model Monitor, and champion/challenger infra ───
# Uses the built-in sagemaker-scikit-learn container for serving.
# The model.tar.gz (produced by the retraining pipeline) bundles both
# training_result.json and inference.py. The Inference Toolkit loads
# inference.py automatically from the archive.

data "aws_sagemaker_prebuilt_ecr_image" "sklearn" {
  repository_name = "sagemaker-scikit-learn"
  image_tag       = "1.2-1-cpu-py3"
}

data "aws_sagemaker_prebuilt_ecr_image" "model_monitor" {
  repository_name = "sagemaker-model-monitor-analyzer"
}

locals {
  sagemaker_sklearn_image       = data.aws_sagemaker_prebuilt_ecr_image.sklearn.registry_path
  sagemaker_model_monitor_image = data.aws_sagemaker_prebuilt_ecr_image.model_monitor.registry_path
  champion_model_name     = "${local.name_prefix}-step-classifier-champion"
  challenger_model_name   = "${local.name_prefix}-step-classifier-challenger"
  endpoint_name           = "${local.name_prefix}-step-classifier"
}

# ── IAM role for SageMaker endpoint + monitoring + Lambda ──────────────

data "aws_iam_policy_document" "sagemaker_endpoint_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_endpoint_execution" {
  name               = "${local.name_prefix}-sagemaker-endpoint-exec"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_endpoint_assume_role.json
}

data "aws_iam_policy_document" "sagemaker_endpoint_execution" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.models.arn,
      "${aws_s3_bucket.models.arn}/*",
    ]
  }
  statement {
    actions = [
      "sagemaker:InvokeEndpoint",
      "sagemaker:DescribeEndpoint",
      "sagemaker:UpdateEndpoint",
      "sagemaker:UpdateEndpointWeightsAndCapacities",
      "sagemaker:CreateEndpointConfig",
      "sagemaker:DescribeEndpointConfig",
      "sagemaker:CreateModel",
      "sagemaker:DeleteModel",
      "sagemaker:DescribeModel",
    ]
    resources = ["*"]
  }
  statement {
    actions = [
      "sagemaker:ListModelPackages",
      "sagemaker:DescribeModelPackage",
      "sagemaker:UpdateModelPackage",
    ]
    resources = ["*"]
  }
  statement {
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
  }
  statement {
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.aws_region}:*:log-group:/aws/lambda/*"]
  }
  # SSM Parameter Store (rollback tracks previous champion)
  statement {
    actions   = ["ssm:GetParameter", "ssm:PutParameter"]
    resources = ["arn:aws:ssm:${var.aws_region}:*:parameter/${local.name_prefix}/champion/*"]
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

resource "aws_iam_policy" "sagemaker_endpoint_execution" {
  name   = "${local.name_prefix}-sagemaker-endpoint-exec"
  policy = data.aws_iam_policy_document.sagemaker_endpoint_execution.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_endpoint_execution" {
  role       = aws_iam_role.sagemaker_endpoint_execution.name
  policy_arn = aws_iam_policy.sagemaker_endpoint_execution.arn
}

resource "aws_iam_role_policy_attachment" "sagemaker_endpoint_execution_managed" {
  role       = aws_iam_role.sagemaker_endpoint_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# ── SageMaker Model (champion) ──────────────────────────────────────────

resource "aws_sagemaker_model" "champion" {
  name = local.champion_model_name

  primary_container {
    image          = local.sagemaker_sklearn_image
    model_data_url = "s3://${aws_s3_bucket.models.bucket}/artifacts/model.tar.gz"
    environment = {
      # SAGEMAKER_SUBMIT_DIRECTORY points the container at a sourcedir.tar.gz
      # containing inference.py + setup.py. The container downloads and extracts
      # it into /opt/ml/code, then pip-installs and imports the module.
      # This is the correct mechanism, model.tar.gz holds model artifacts only.
      SAGEMAKER_PROGRAM          = "inference"
      SAGEMAKER_SUBMIT_DIRECTORY = "s3://${aws_s3_bucket.models.bucket}/artifacts/sourcedir.tar.gz"
    }
  }

  execution_role_arn = aws_iam_role.sagemaker_endpoint_execution.arn

  tags = { ModelType = "champion" }

  depends_on = [
    aws_iam_role.sagemaker_endpoint_execution,
    null_resource.bootstrap_artifact,
  ]
}

# ── SageMaker Model (challenger) ────────────────────────────────────────

resource "aws_sagemaker_model" "challenger" {
  name = local.challenger_model_name

  primary_container {
    image          = local.sagemaker_sklearn_image
    model_data_url = "s3://${aws_s3_bucket.models.bucket}/artifacts/model.tar.gz"
    environment = {
      SAGEMAKER_PROGRAM          = "inference"
      SAGEMAKER_SUBMIT_DIRECTORY = "s3://${aws_s3_bucket.models.bucket}/artifacts/sourcedir.tar.gz"
    }
  }

  execution_role_arn = aws_iam_role.sagemaker_endpoint_execution.arn

  tags = { ModelType = "challenger" }

  depends_on = [null_resource.bootstrap_artifact]
}

# ── Endpoint config (champion 90%, challenger 10% canary) ───────────────

resource "aws_sagemaker_endpoint_configuration" "step_classifier" {
  name_prefix = "${local.name_prefix}-step-clf-config-"

  production_variants {
    variant_name           = "champion"
    model_name             = aws_sagemaker_model.champion.name
    initial_instance_count = 1
    instance_type          = var.sagemaker_endpoint_instance_type
    initial_variant_weight = 90
    container_startup_health_check_timeout_in_seconds = 60
  }

  production_variants {
    variant_name           = "challenger"
    model_name             = aws_sagemaker_model.challenger.name
    initial_instance_count = 1
    instance_type          = var.sagemaker_endpoint_instance_type
    initial_variant_weight = 10
    container_startup_health_check_timeout_in_seconds = 60
  }

  data_capture_config {
    enable_capture              = true
    initial_sampling_percentage = 100
    destination_s3_uri          = "s3://${aws_s3_bucket.models.bucket}/data-capture"

    capture_options { capture_mode = "Input" }
    capture_options { capture_mode = "Output" }
  }

  lifecycle { create_before_destroy = true }
}

# ── SageMaker endpoint ──────────────────────────────────────────────────

resource "aws_sagemaker_endpoint" "step_classifier" {
  name                 = local.endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.step_classifier.name

  tags = { Name = local.endpoint_name }

  depends_on = [aws_sagemaker_endpoint_configuration.step_classifier]
}

# ── Model Monitor baseline ──────────────────────────────────────────────
# Computed by the pipeline's ComputeMonitorBaseline step.

locals {
  monitor_baseline_uri = "s3://${aws_s3_bucket.models.bucket}/monitor-baseline"
}

# ── Model Monitor: data drift schedule (hourly) ────────────────────────

resource "aws_sagemaker_monitoring_schedule" "data_drift" {
  name = "${local.name_prefix}-data-drift-monitor"

  monitoring_schedule_config {
    monitoring_job_definition_name = aws_sagemaker_data_quality_job_definition.data_drift.name
    monitoring_type                = "DataQuality"

    schedule_config {
      schedule_expression = "cron(0 * ? * * *)"
    }
  }

  depends_on = [
    aws_sagemaker_data_quality_job_definition.data_drift,
    aws_sagemaker_endpoint.step_classifier,
  ]
}

resource "aws_sagemaker_data_quality_job_definition" "data_drift" {
  name = "${local.name_prefix}-data-drift-def"

  role_arn = aws_iam_role.sagemaker_endpoint_execution.arn

  data_quality_app_specification {
    image_uri = local.sagemaker_model_monitor_image
  }

  data_quality_baseline_config {
    constraints_resource { s3_uri = "${local.monitor_baseline_uri}/constraints.json" }
    statistics_resource  { s3_uri = "${local.monitor_baseline_uri}/statistics.json" }
  }

  data_quality_job_input {
    endpoint_input {
      endpoint_name = aws_sagemaker_endpoint.step_classifier.name
      local_path    = "/opt/ml/processing/input"
      s3_input_mode = "File"
    }
  }

  data_quality_job_output_config {
    monitoring_outputs {
      s3_output {
        s3_uri          = "s3://${aws_s3_bucket.models.bucket}/monitor-results"
        local_path      = "/opt/ml/processing/output"
        s3_upload_mode  = "EndOfJob"
      }
    }
  }

  job_resources {
    cluster_config {
      instance_count    = 1
      instance_type     = "ml.t3.medium"
      volume_size_in_gb = 20
    }
  }

  stopping_condition {
    max_runtime_in_seconds = 3600
  }
}

# ── Champion/challenger promotion Lambda ────────────────────────────────
# Triggered by EventBridge when a new model package is created.

resource "aws_lambda_function" "promote_champion" {
  filename         = "${path.module}/promote_lambda_payload.zip"
  source_code_hash = filebase64sha256("${path.module}/promote_champion.py")
  function_name    = "${local.name_prefix}-promote-champion"
  role             = aws_iam_role.sagemaker_endpoint_execution.arn
  handler          = "promote_champion.lambda_handler"
  runtime          = "python3.12"
  timeout          = 120
  memory_size      = 256
  depends_on       = [null_resource.promote_lambda_source]

  environment {
    variables = {
      MODEL_PACKAGE_GROUP    = aws_sagemaker_model_package_group.models.model_package_group_name
      ENDPOINT_NAME          = aws_sagemaker_endpoint.step_classifier.name
      PREFIX                 = local.name_prefix
      MODEL_BUCKET           = aws_s3_bucket.models.bucket
      SAGEMAKER_SKLEARN_IMAGE = local.sagemaker_sklearn_image
      EXECUTION_ROLE_ARN     = aws_iam_role.sagemaker_endpoint_execution.arn
    }
  }
}

resource "aws_lambda_permission" "promote_champion" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.promote_champion.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.model_registered.arn
}

resource "aws_cloudwatch_event_rule" "model_registered" {
  name        = "${local.name_prefix}-model-registered"
  description = "Fires when a new model package is created. Lambda auto-approves and promotes"

  event_pattern = jsonencode({
    source      = ["aws.sagemaker"]
    detail-type = ["SageMaker Model Package State Change"]
    detail = {
      ModelPackageGroupName = [aws_sagemaker_model_package_group.models.model_package_group_name]
      ModelApprovalStatus   = ["PendingManualApproval"]
    }
  })
}

resource "aws_cloudwatch_event_target" "promote_champion" {
  rule      = aws_cloudwatch_event_rule.model_registered.name
  arn       = aws_lambda_function.promote_champion.arn
  target_id = "PromoteChampion"
}

# ── Rollback Lambda (triggered by drift/error alarms) ───────────────────
# Reverts the endpoint to the previous champion version.

resource "aws_lambda_function" "rollback" {
  filename         = "${path.module}/rollback_lambda_payload.zip"
  source_code_hash = filebase64sha256("${path.module}/rollback.py")
  function_name    = "${local.name_prefix}-rollback-champion"
  role             = aws_iam_role.sagemaker_endpoint_execution.arn
  handler          = "rollback.lambda_handler"
  runtime          = "python3.12"
  timeout          = 120
  memory_size      = 256
  depends_on       = [null_resource.rollback_lambda_source]

  environment {
    variables = {
      ENDPOINT_NAME          = aws_sagemaker_endpoint.step_classifier.name
      PREFIX                 = local.name_prefix
      MODEL_PACKAGE_GROUP    = aws_sagemaker_model_package_group.models.model_package_group_name
      SAGEMAKER_SKLEARN_IMAGE = local.sagemaker_sklearn_image
      EXECUTION_ROLE_ARN     = aws_iam_role.sagemaker_endpoint_execution.arn
      MODEL_BUCKET           = aws_s3_bucket.models.bucket
    }
  }
}

resource "aws_lambda_permission" "rollback" {
  statement_id  = "AllowSNSTriggerRollback"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.rollback.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.drift_alarm.arn
}

resource "aws_sns_topic_subscription" "rollback_subscription" {
  topic_arn = aws_sns_topic.drift_alarm.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.rollback.arn
}

# ── CloudWatch alarms for drift and endpoint health ─────────────────────

resource "aws_cloudwatch_metric_alarm" "data_drift_high" {
  alarm_name          = "${local.name_prefix}-data-drift-high"
  alarm_description   = "Data drift detected on the step classifier endpoint, rollback triggered."
  namespace           = "AWS/SageMaker"
  metric_name         = "DriftViolationCount"
  dimensions = {
    MonitoringSchedule = aws_sagemaker_monitoring_schedule.data_drift.name
  }
  statistic           = "Sum"
  period              = 3600
  evaluation_periods  = 1
  threshold           = var.data_drift_violation_threshold
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.drift_alarm.arn]
  ok_actions          = [aws_sns_topic.drift_alarm.arn]
}

resource "aws_cloudwatch_metric_alarm" "endpoint_error_rate" {
  alarm_name          = "${local.name_prefix}-endpoint-4xx-rate"
  alarm_description   = "High 4XX error rate on the step classifier endpoint."
  namespace           = "AWS/SageMaker"
  metric_name         = "ModelInvocation4XXErrors"
  dimensions = {
    EndpointName = aws_sagemaker_endpoint.step_classifier.name
    VariantName  = "champion"
  }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 10
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.drift_alarm.arn]
  ok_actions          = [aws_sns_topic.drift_alarm.arn]
}

resource "aws_cloudwatch_metric_alarm" "endpoint_latency" {
  alarm_name          = "${local.name_prefix}-endpoint-latency"
  alarm_description   = "High p99 latency on the step classifier endpoint."
  namespace           = "AWS/SageMaker"
  metric_name         = "ModelLatency"
  dimensions = {
    EndpointName = aws_sagemaker_endpoint.step_classifier.name
    VariantName  = "champion"
  }
  extended_statistic  = "p99"
  period              = 300
  evaluation_periods  = 2
  threshold           = 2000
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.drift_alarm.arn]
  ok_actions          = [aws_sns_topic.drift_alarm.arn]
}

resource "aws_sns_topic" "drift_alarm" {
  name = "${local.name_prefix}-drift-alarm"
}

resource "aws_sns_topic_subscription" "drift_alarm_email" {
  count     = var.drift_alarm_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.drift_alarm.arn
  protocol  = "email"
  endpoint  = var.drift_alarm_email
}

# ── Bootstrap placeholder artifact ──────────────────────────────────────
# Before the first pipeline run, the S3 artifact path is empty.  Create
# a minimal model.tar.gz so the endpoint can deploy.

resource "null_resource" "bootstrap_artifact" {
  triggers = {
    bucket_name    = aws_s3_bucket.models.bucket
    # Bump this string whenever the inline inference script changes
    # so Terraform re-uploads both artifacts with the new content.
    script_version = "v3-submit-directory"
  }

  provisioner "local-exec" {
    command = <<-EOC
      TMP=$(mktemp -d)

      # model.tar.gz, model artifacts only (no inference code)
      echo '{"model_id":"step_classifier_v1","version":"bootstrap"}' > "$TMP/training_result.json"
      tar -C "$TMP" -czf "$TMP/model.tar.gz" training_result.json
      aws s3 cp "$TMP/model.tar.gz" "s3://${aws_s3_bucket.models.bucket}/artifacts/model.tar.gz"

      # sourcedir.tar.gz, inference code only.
      # SAGEMAKER_SUBMIT_DIRECTORY points the container here; it downloads,
      # extracts to /opt/ml/code, then pip-installs and imports the module.
      cat > "$TMP/inference.py" << 'PYEOF'
import json
import logging
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    return {"model_id":"step_classifier_v1","version":"bootstrap"}

def input_fn(body, content_type="application/json"):
    print(f"DEBUG input_fn: type={type(body).__name__} content_type={content_type}")
    if isinstance(body, bytes):
        print(f"DEBUG input_fn: hex={body[:100].hex()}")
    if content_type == "application/json":
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    raise ValueError(f"Unsupported content_type: {content_type}")

def predict_fn(data, model):
    return {"predicted_outcome":"completed","confidence":0.95,"model_id":model.get("model_id","unknown"),"model_version":model.get("version",0)}

def output_fn(pred, accept="application/json"):
    return json.dumps(pred)
PYEOF
      cat > "$TMP/setup.py" << 'PYEOF'
from setuptools import setup
setup(name="inference", version="1.0.0", py_modules=["inference"])
PYEOF
      tar -C "$TMP" -czf "$TMP/sourcedir.tar.gz" inference.py setup.py
      aws s3 cp "$TMP/sourcedir.tar.gz" "s3://${aws_s3_bucket.models.bucket}/artifacts/sourcedir.tar.gz"

      rm -rf "$TMP"
    EOC
  }
}

# ── Lambda source zips ──────────────────────────────────────────────────

resource "null_resource" "promote_lambda_source" {
  triggers = {
    script_hash = filemd5("${path.module}/promote_champion.py")
  }
  provisioner "local-exec" {
    command = "zip -j ${path.module}/promote_lambda_payload.zip ${path.module}/promote_champion.py"
  }
}

resource "null_resource" "rollback_lambda_source" {
  triggers = {
    script_hash = filemd5("${path.module}/rollback.py")
  }
  provisioner "local-exec" {
    command = "zip -j ${path.module}/rollback_lambda_payload.zip ${path.module}/rollback.py"
  }
}
