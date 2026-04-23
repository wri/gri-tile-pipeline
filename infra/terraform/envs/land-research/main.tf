module "state_usw2" {
  source = "../../modules/lithops-prereqs"
  region = "us-west-2"
  tags   = var.tags
}

module "state_euc1" {
  source = "../../modules/lithops-prereqs"
  providers = {
    aws = aws.euc1
  }
  region = "eu-central-1"
  tags   = var.tags
}

module "state_use1" {
  source = "../../modules/lithops-prereqs"
  providers = {
    aws = aws.use1
  }
  region = "us-east-1"
  tags   = var.tags
}

module "role" {
  source = "../../modules/lithops-iam-role"
  name   = var.role_name
  lithops_state_bucket_arns = [
    module.state_usw2.bucket_arn,
    module.state_euc1.bucket_arn,
    module.state_use1.bucket_arn,
  ]
  output_bucket_arn = aws_s3_bucket.ttc_data.arn
  tags              = var.tags
}

# Dedicated TTC data bucket (ARD + predictions), co-located with the predict
# Lambda in us-east-1. Intelligent-Tiering via a day-0 lifecycle rule means
# uploads land in Standard and AWS auto-manages access-tier placement from
# there — callers don't have to specify a storage class.
resource "aws_s3_bucket" "ttc_data" {
  provider = aws.use1
  bucket   = var.output_bucket_name
  tags     = var.tags
}

resource "aws_s3_bucket_public_access_block" "ttc_data" {
  provider                = aws.use1
  bucket                  = aws_s3_bucket.ttc_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "ttc_data" {
  provider = aws.use1
  bucket   = aws_s3_bucket.ttc_data.id

  rule {
    id     = "intelligent-tiering"
    status = "Enabled"
    filter {}

    transition {
      days          = 0
      storage_class = "INTELLIGENT_TIERING"
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}
