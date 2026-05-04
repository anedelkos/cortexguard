resource "aws_cloudwatch_log_group" "cloud_api" {
  name              = "/ecs/${local.name_prefix}-cloud-api"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/ecs/${local.name_prefix}-worker"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "qdrant" {
  name              = "/ecs/${local.name_prefix}-qdrant"
  retention_in_days = var.log_retention_days
}

# ── DLQ alarm ─────────────────────────────────────────────────────────────────
# Messages in the DLQ mean the worker failed to process a mayday escalation 3 times.

resource "aws_sns_topic" "dlq_alarm" {
  name = "${local.name_prefix}-dlq-alarm"
}

resource "aws_sns_topic_subscription" "dlq_alarm_email" {
  count     = var.dlq_alarm_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.dlq_alarm.arn
  protocol  = "email"
  endpoint  = var.dlq_alarm_email
}

resource "aws_cloudwatch_metric_alarm" "dlq_not_empty" {
  alarm_name          = "${local.name_prefix}-dlq-not-empty"
  alarm_description   = "Mayday messages in the DLQ indicate repeated planning failures — investigate immediately."
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  dimensions          = { QueueName = aws_sqs_queue.mayday_dlq.name }
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.dlq_alarm.arn]
  ok_actions          = [aws_sns_topic.dlq_alarm.arn]
}
