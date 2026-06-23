resource "aws_sqs_queue" "mayday_dlq" {
  name                       = "${local.name_prefix}-mayday-dlq"
  message_retention_seconds  = 1209600  # 14 days
  sqs_managed_sse_enabled    = true
}

resource "aws_sqs_queue" "mayday" {
  name                       = "${local.name_prefix}-mayday"
  visibility_timeout_seconds = 300  # 5 min: must be >= worker processing time
  message_retention_seconds  = 86400
  sqs_managed_sse_enabled    = true

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.mayday_dlq.arn
    maxReceiveCount     = 3
  })
}
