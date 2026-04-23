terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Output-only module: generates the IAM policy statements that would grant
# `grantee_role_arns` cross-account read/write on `bucket_name` (optionally
# scoped to `key_prefixes`). It does NOT apply a bucket policy

locals {
  object_resources = length(var.key_prefixes) == 0 ? ["arn:aws:s3:::${var.bucket_name}/*"] : [
    for p in var.key_prefixes : "arn:aws:s3:::${var.bucket_name}/${trimsuffix(p, "/")}/*"
  ]
}

data "aws_iam_policy_document" "access" {
  statement {
    sid    = "CrossAccountListBucket"
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = var.grantee_role_arns
    }
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = ["arn:aws:s3:::${var.bucket_name}"]

    dynamic "condition" {
      for_each = length(var.key_prefixes) == 0 ? [] : [1]
      content {
        test     = "StringLike"
        variable = "s3:prefix"
        values   = [for p in var.key_prefixes : "${trimsuffix(p, "/")}/*"]
      }
    }
  }

  statement {
    sid    = "CrossAccountReadWrite"
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = var.grantee_role_arns
    }
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = local.object_resources
  }
}
