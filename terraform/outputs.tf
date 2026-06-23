output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.cloud_api.dns_name
}

output "alb_base_url" {
  description = "Base URL for the cloud API: set as CLOUD_API_URL on the edge"
  value       = "${var.acm_certificate_arn != "" ? "https" : "http"}://${aws_lb.cloud_api.dns_name}"
}

output "ecr_repository_url" {
  description = "ECR repository URL: use this in docker push and as var.image_tag prefix"
  value       = aws_ecr_repository.cloud_api.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.cloud_api.name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for cloud-api container logs"
  value       = aws_cloudwatch_log_group.cloud_api.name
}

output "sqs_queue_url" {
  description = "SQS planning queue URL: set as CLOUD_SQS_QUEUE_URL on both cloud-api and worker"
  value       = aws_sqs_queue.mayday.url
}

output "sqs_dlq_url" {
  description = "SQS dead-letter queue URL: inspect for failed planning messages"
  value       = aws_sqs_queue.mayday_dlq.url
}

output "rds_endpoint" {
  description = "RDS Postgres endpoint (host:port)"
  value       = "${aws_db_instance.main.address}:${aws_db_instance.main.port}"
}

output "qdrant_url" {
  description = "Qdrant URL (via Cloud Map DNS, reachable from ECS tasks only)"
  value       = local.qdrant_url
}

# --- Retraining pipeline outputs ---

output "model_bucket_name" {
  description = "S3 bucket for model artifacts and retraining source code"
  value       = aws_s3_bucket.models.bucket
}

output "model_registry_name" {
  description = "SageMaker Model Package Group name for edge step-classifier models"
  value       = aws_sagemaker_model_package_group.models.model_package_group_name
}

output "pipeline_name" {
  description = "SageMaker retraining pipeline name"
  value       = aws_sagemaker_pipeline.retraining.pipeline_name
}

# --- Endpoint outputs ---

output "endpoint_name" {
  description = "SageMaker endpoint name for the step classifier"
  value       = aws_sagemaker_endpoint.step_classifier.name
}

output "endpoint_arn" {
  description = "SageMaker endpoint ARN"
  value       = aws_sagemaker_endpoint.step_classifier.arn
}

output "monitoring_schedule_name" {
  description = "SageMaker Model Monitor schedule name for data drift"
  value       = aws_sagemaker_monitoring_schedule.data_drift.name
}

output "drift_alarm_topic_arn" {
  description = "SNS topic ARN for drift and endpoint health alarms"
  value       = aws_sns_topic.drift_alarm.arn
}

output "promote_lambda_name" {
  description = "Lambda function name for champion/challenger promotion"
  value       = aws_lambda_function.promote_champion.function_name
}

output "rollback_lambda_name" {
  description = "Lambda function name for rollback on drift/error"
  value       = aws_lambda_function.rollback.function_name
}
