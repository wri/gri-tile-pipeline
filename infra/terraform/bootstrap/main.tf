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
  region = var.region
  default_tags {
    tags = {
      "wri:project" = "gri-ttc-tile-generator"
      "ManagedBy"   = "terraform"
    }
  }
}

variable "region" {
  description = "AWS region for the Terraform state backend."
  type        = string
  default     = "us-west-2"
}

variable "state_bucket_name" {
  description = "S3 bucket that will hold Terraform state for all envs."
  type        = string
  default     = "gri-tile-pipeline-tfstate"
}

variable "lock_table_name" {
  description = "DynamoDB table for Terraform state locking."
  type        = string
  default     = "gri-tile-pipeline-tflock"
}

resource "aws_s3_bucket" "tfstate" {
  bucket = var.state_bucket_name
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "tflock" {
  name         = var.lock_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

output "state_bucket" {
  value = aws_s3_bucket.tfstate.id
}

output "lock_table" {
  value = aws_dynamodb_table.tflock.id
}

output "backend_snippet" {
  description = "Paste this into envs/<env>/main.tf terraform { backend \"s3\" { ... } }"
  value       = <<-EOT
    backend "s3" {
      bucket         = "${aws_s3_bucket.tfstate.id}"
      key            = "envs/<env>/terraform.tfstate"
      region         = "${var.region}"
      dynamodb_table = "${aws_dynamodb_table.tflock.id}"
      encrypt        = true
    }
  EOT
}
