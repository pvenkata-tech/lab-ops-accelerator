terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── VPC ───────────────────────────────────────────────────────────────────────
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ── RDS PostgreSQL ────────────────────────────────────────────────────────────
resource "aws_db_instance" "postgres" {
  identifier        = "${var.app_name}-postgres"
  engine            = "postgres"
  engine_version    = "16"
  instance_class    = var.db_instance_class
  allocated_storage = 20
  db_name           = "labops"
  username          = "labops"
  password          = random_password.db.result

  storage_encrypted      = true
  deletion_protection    = true
  skip_final_snapshot    = false
  final_snapshot_identifier = "${var.app_name}-final-snapshot"

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  tags = local.tags
}

resource "random_password" "db" {
  length  = 32
  special = false
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.app_name}-db-subnet"
  subnet_ids = data.aws_subnets.private.ids
  tags       = local.tags
}

# ── Secrets Manager ───────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "app_secrets" {
  name = "${var.app_name}/secrets"
  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  secret_string = jsonencode({
    DATABASE_URL            = "postgresql+asyncpg://labops:${random_password.db.result}@${aws_db_instance.postgres.endpoint}/labops"
    CHECKPOINT_DATABASE_URL = "postgresql://labops:${random_password.db.result}@${aws_db_instance.postgres.endpoint}/labops"
    LIMS_API_KEY            = var.lims_api_key
    EHR_API_KEY             = var.ehr_api_key
    LANGCHAIN_API_KEY       = var.langchain_api_key
  })
}

# ── ECS Cluster ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = var.app_name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.tags
}

# ── IAM Task Role ─────────────────────────────────────────────────────────────
resource "aws_iam_role" "task" {
  name = "${var.app_name}-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "bedrock" {
  name = "bedrock-access"
  role = aws_iam_role.task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["bedrock:InvokeModel"]
        Resource = [
          "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.bedrock_claude_model_id}",
          "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.bedrock_embedding_model_id}",
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [aws_secretsmanager_secret.app_secrets.arn]
      }
    ]
  })
}

# ── ECS Task Definition ───────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "accelerator" {
  family                   = var.app_name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name  = "accelerator"
    image = var.container_image
    portMappings = [{ containerPort = 8000 }]
    secrets = [
      { name = "DATABASE_URL",            valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:DATABASE_URL::" },
      { name = "CHECKPOINT_DATABASE_URL", valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:CHECKPOINT_DATABASE_URL::" },
      { name = "LIMS_API_KEY",            valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:LIMS_API_KEY::" },
      { name = "EHR_API_KEY",             valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:EHR_API_KEY::" },
      { name = "LANGCHAIN_API_KEY",       valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:LANGCHAIN_API_KEY::" },
    ]
    environment = [
      { name = "AWS_REGION",                  value = var.aws_region },
      { name = "BEDROCK_CLAUDE_MODEL_ID",     value = var.bedrock_claude_model_id },
      { name = "BEDROCK_EMBEDDING_MODEL_ID",  value = var.bedrock_embedding_model_id },
      { name = "LIMS_API_BASE_URL",           value = var.lims_api_base_url },
      { name = "EHR_WEBHOOK_URL",             value = var.ehr_webhook_url },
      { name = "HITL_CONFIDENCE_THRESHOLD",   value = tostring(var.hitl_confidence_threshold) },
      { name = "LANGCHAIN_TRACING_V2",        value = "true" },
      { name = "LANGSMITH_PROJECT",           value = "lab-ops-accelerator-prod" },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.accelerator.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "accelerator"
      }
    }
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])

  tags = local.tags
}

resource "aws_iam_role" "execution" {
  name = "${var.app_name}-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ── ECS Service ───────────────────────────────────────────────────────────────
resource "aws_ecs_service" "accelerator" {
  name            = var.app_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.accelerator.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.private.ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.accelerator.arn
    container_name   = "accelerator"
    container_port   = 8000
  }

  tags = local.tags
}

# ── ALB ───────────────────────────────────────────────────────────────────────
resource "aws_lb" "accelerator" {
  name               = var.app_name
  internal           = true
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.private.ids
  tags               = local.tags
}

resource "aws_lb_target_group" "accelerator" {
  name        = var.app_name
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.accelerator.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.accelerator.arn
  }
}

# ── Security Groups ───────────────────────────────────────────────────────────
resource "aws_security_group" "alb" {
  name   = "${var.app_name}-alb"
  vpc_id = data.aws_vpc.default.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

resource "aws_security_group" "ecs" {
  name   = "${var.app_name}-ecs"
  vpc_id = data.aws_vpc.default.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

resource "aws_security_group" "rds" {
  name   = "${var.app_name}-rds"
  vpc_id = data.aws_vpc.default.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  tags = local.tags
}

# ── CloudWatch ────────────────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "accelerator" {
  name              = "/ecs/${var.app_name}"
  retention_in_days = 30
  tags              = local.tags
}

locals {
  tags = {
    Project     = var.app_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
