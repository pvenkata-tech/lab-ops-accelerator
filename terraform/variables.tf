variable "app_name" {
  description = "Application name used for resource naming"
  type        = string
  default     = "lab-ops-accelerator"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "prod"
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "container_image" {
  description = "Docker image URI (ECR) for the accelerator service"
  type        = string
}

variable "bedrock_claude_model_id" {
  description = "Bedrock model ID for Claude inference"
  type        = string
  default     = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
}

variable "bedrock_embedding_model_id" {
  description = "Bedrock model ID for Titan embeddings"
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "task_cpu" {
  description = "ECS task CPU units"
  type        = number
  default     = 1024
}

variable "task_memory" {
  description = "ECS task memory (MB)"
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Desired ECS task count"
  type        = number
  default     = 2
}

variable "hitl_confidence_threshold" {
  description = "Confidence below which cases are routed to HITL"
  type        = number
  default     = 0.80
}

variable "lims_api_base_url" {
  description = "Base URL for the LIMS API"
  type        = string
}

variable "ehr_webhook_url" {
  description = "EHR webhook URL for physician notifications"
  type        = string
}

variable "lims_api_key" {
  description = "API key for LIMS"
  type        = string
  sensitive   = true
}

variable "ehr_api_key" {
  description = "API key for EHR notifications"
  type        = string
  sensitive   = true
}

variable "langchain_api_key" {
  description = "LangSmith API key for tracing"
  type        = string
  sensitive   = true
  default     = ""
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for ALB HTTPS listener"
  type        = string
}

variable "allowed_cidr" {
  description = "CIDR allowed to reach the ALB (internal network)"
  type        = string
  default     = "10.0.0.0/8"
}
