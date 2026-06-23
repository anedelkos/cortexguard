resource "random_password" "db_password" {
  length  = 32
  special = false  # avoid DSN-unsafe characters
}

resource "aws_db_subnet_group" "main" {
  name       = "${local.name_prefix}-db"
  subnet_ids = data.aws_subnets.selected.ids
}

resource "aws_security_group" "rds" {
  name        = "${local.name_prefix}-rds"
  description = "Allow Postgres from ECS tasks"
  vpc_id      = data.aws_vpc.selected.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [
      aws_security_group.ecs_tasks.id,
      aws_security_group.worker.id,
      aws_security_group.sagemaker.id,
    ]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "main" {
  identifier        = "${local.name_prefix}-db"
  engine            = "postgres"
  engine_version    = "16"
  instance_class    = "db.t3.micro"  # free tier eligible (750h/month for 12 months)
  allocated_storage = 20

  db_name  = "cortexguard"
  username = "cortexguard"
  password = random_password.db_password.result

  db_subnet_group_name      = aws_db_subnet_group.main.name
  vpc_security_group_ids    = [aws_security_group.rds.id]
  publicly_accessible       = false
  multi_az                  = false
  skip_final_snapshot       = true
  final_snapshot_identifier = null
  storage_type              = "gp3"
  backup_retention_period   = 7
  deletion_protection       = false
}

locals {
  db_url = "postgresql://cortexguard:${random_password.db_password.result}@${aws_db_instance.main.address}:5432/cortexguard"
}
