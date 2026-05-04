variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "eu-west-1"
}

variable "project" {
  description = "Project name — used as a prefix on all resource names"
  type        = string
  default     = "cortexguard"
}

variable "environment" {
  description = "Deployment environment tag (e.g. prod, staging, demo)"
  type        = string
  default     = "demo"
}

# --- Container image ---

variable "image_tag" {
  description = "Docker image tag to deploy. Must be a specific tag in non-demo environments — 'latest' is rejected."
  type        = string
  default     = "latest"
}

variable "qdrant_image_tag" {
  description = "Qdrant image tag to deploy. Must be pinned to a specific semver (e.g. v1.9.4) — floating tags risk silent schema breakage against persisted EFS data."
  type        = string
}

# --- Networking ---

variable "vpc_id" {
  description = "VPC to deploy into. Leave empty to use the default VPC."
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnet IDs for the ECS tasks and ALB. Leave empty to discover from the default VPC."
  type        = list(string)
  default     = []
}

# --- ECS task sizing ---

variable "task_cpu" {
  description = "Fargate task CPU units (256 = 0.25 vCPU)"
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Fargate task memory in MiB"
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Number of cloud-api ECS task replicas"
  type        = number
  default     = 1
}

variable "worker_count" {
  description = "Number of SQS planning worker replicas"
  type        = number
  default     = 1
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for HTTPS. Leave empty for HTTP-only (demo deployments in private networks only)."
  type        = string
  default     = ""
}

variable "assign_public_ip" {
  description = "Assign public IPs to Fargate tasks. Required when tasks run in public subnets without a NAT gateway (the default VPC). Set to false when deploying into private subnets with a NAT gateway or VPC endpoints."
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days for all ECS log groups."
  type        = number
  default     = 7
}

variable "dlq_alarm_email" {
  description = "Email address to notify when messages land in the mayday DLQ. Leave empty to create the SNS topic without a subscription."
  type        = string
  default     = ""
}

# --- Application secrets (stored in Secrets Manager) ---

variable "groq_api_key" {
  description = "Groq API key for cloud LLM calls"
  type        = string
  sensitive   = true
  default     = ""
}

variable "anthropic_api_key" {
  description = "Anthropic API key (optional — leave empty if using Groq)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "cloud_api_key" {
  description = "Shared-secret API key the edge must send in X-CortexGuard-Key header"
  type        = string
  sensitive   = true
}

# --- LLM backend ---

variable "llm_backend" {
  description = "LLM backend to use: groq | anthropic | mock"
  type        = string
  default     = "groq"
}

# Cross-variable validations (requires Terraform >= 1.6).
# triggers_replace forces resource replacement (and precondition re-evaluation) whenever
# these values change. Sensitive variables (groq_api_key, anthropic_api_key, cloud_api_key)
# are intentionally excluded — Terraform stores triggers_replace values in state in plaintext,
# and preconditions referencing those variables already run on every apply without needing a trigger.
resource "terraform_data" "input_validation" {
  triggers_replace = [
    var.image_tag,
    var.qdrant_image_tag,
    var.environment,
    var.llm_backend,
    var.acm_certificate_arn,
  ]

  lifecycle {
    precondition {
      condition     = !(var.llm_backend == "groq" && var.groq_api_key == "")
      error_message = "groq_api_key is required when llm_backend is 'groq'."
    }
    precondition {
      condition     = !(var.llm_backend == "anthropic" && var.anthropic_api_key == "")
      error_message = "anthropic_api_key is required when llm_backend is 'anthropic'."
    }
    precondition {
      condition     = !(var.acm_certificate_arn == "" && var.environment != "demo")
      error_message = "acm_certificate_arn is required for non-demo environments — the API key is sent as a plain HTTP header otherwise."
    }
    precondition {
      condition     = !(var.image_tag == "latest" && var.environment != "demo")
      error_message = "image_tag must be pinned to a specific tag in non-demo environments."
    }
    precondition {
      condition     = var.qdrant_image_tag != "latest"
      error_message = "qdrant_image_tag must be a pinned semver (e.g. v1.9.4) — 'latest' risks silent schema breakage against persisted EFS data."
    }
    precondition {
      condition     = length(var.cloud_api_key) >= 32
      error_message = "cloud_api_key must be at least 32 characters — use a random secret, not the example placeholder."
    }
  }
}
