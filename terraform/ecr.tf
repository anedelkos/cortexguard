resource "aws_ecr_repository" "cloud_api" {
  name                 = "${local.name_prefix}-cloud-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Lifecycle policy: keep the 5 most recent tagged images; expire untagged layers after 1 day.
# Two separate rules prevent the "any" selector from expiring a tagged image that is still
# referenced by a running ECS task definition.
resource "aws_ecr_lifecycle_policy" "cloud_api" {
  repository = aws_ecr_repository.cloud_api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged layers after 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      }
    ]
  })
}
