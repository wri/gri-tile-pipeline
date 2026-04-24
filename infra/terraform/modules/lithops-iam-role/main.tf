terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

data "aws_iam_policy_document" "assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "this" {
  name               = var.name
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
  tags               = var.tags
}

# CloudWatch Logs (AWS-managed basic execution policy).
resource "aws_iam_role_policy_attachment" "basic_execution" {
  role       = aws_iam_role.this.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Lithops state buckets (futures, args, task logs) + the pipeline output bucket.
data "aws_iam_policy_document" "inline" {
  statement {
    sid     = "LithopsStateBuckets"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket", "s3:GetBucketLocation"]
    resources = concat(
      var.lithops_state_bucket_arns,
      [for arn in var.lithops_state_bucket_arns : "${arn}/*"],
    )
  }

  statement {
    sid     = "OutputBucket"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:GetBucketLocation"]
    resources = [
      var.output_bucket_arn,
      "${var.output_bucket_arn}/*",
    ]
  }

  # External public/requester-pays buckets the loaders read from.
  # - copernicus-dem-30m: Copernicus DEM GLO-30 via Earth Search (DEM loader).
  # - sentinel-cogs: Sentinel-2 L2A COGs via Earth Search (S2 loader).
  # Add entries here when a new external dataset is introduced.
  statement {
    sid     = "ExternalPublicDataRead"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"]
    resources = [
      "arn:aws:s3:::copernicus-dem-30m",
      "arn:aws:s3:::copernicus-dem-30m/*",
      "arn:aws:s3:::sentinel-cogs",
      "arn:aws:s3:::sentinel-cogs/*",
    ]
  }

  # Lithops pulls images from ECR in the same account. Lithops itself creates
  # the repos via `lithops runtime build`; the execution role only needs pull.
  statement {
    sid    = "EcrPull"
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "inline" {
  name   = "lithops-access"
  role   = aws_iam_role.this.id
  policy = data.aws_iam_policy_document.inline.json
}
