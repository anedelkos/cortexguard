# Store sensitive values in Secrets Manager.
# Each secret is fetched by the ECS task at runtime via valueFrom in the task definition.

locals {
  # Immediate deletion on destroy is acceptable only for demo environments.
  # Non-demo environments use a 7-day recovery window to prevent accidental key loss.
  secret_recovery_window = var.environment == "demo" ? 0 : 7
}

resource "aws_secretsmanager_secret" "groq_api_key" {
  name                    = "${local.name_prefix}/groq-api-key"
  recovery_window_in_days = local.secret_recovery_window
}

resource "aws_secretsmanager_secret_version" "groq_api_key" {
  secret_id     = aws_secretsmanager_secret.groq_api_key.id
  secret_string = var.groq_api_key != "" ? var.groq_api_key : "not-set"
}

resource "aws_secretsmanager_secret" "cloud_api_key" {
  name                    = "${local.name_prefix}/cloud-api-key"
  recovery_window_in_days = local.secret_recovery_window
}

resource "aws_secretsmanager_secret_version" "cloud_api_key" {
  secret_id     = aws_secretsmanager_secret.cloud_api_key.id
  secret_string = var.cloud_api_key
}

resource "aws_secretsmanager_secret" "db_url" {
  name                    = "${local.name_prefix}/db-url"
  recovery_window_in_days = local.secret_recovery_window
}

resource "aws_secretsmanager_secret_version" "db_url" {
  secret_id     = aws_secretsmanager_secret.db_url.id
  secret_string = local.db_url
}

resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name                    = "${local.name_prefix}/anthropic-api-key"
  recovery_window_in_days = local.secret_recovery_window
}

resource "aws_secretsmanager_secret_version" "anthropic_api_key" {
  secret_id     = aws_secretsmanager_secret.anthropic_api_key.id
  secret_string = var.anthropic_api_key != "" ? var.anthropic_api_key : "not-set"
}
